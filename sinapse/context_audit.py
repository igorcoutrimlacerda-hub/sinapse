"""Auditoria reproduzível da seleção de contexto do Farol.

Compara três condições com o mesmo modelo, projeto e pedido:

- ``current``: fluxo atual de ``chat_with_deepseek``;
- ``recent_only``: referência canônica + histórico recente, sem memória;
- ``selective``: pula memória/estado em turnos casuais e injeta estado nos
  turnos que pedem informação.

O cliente é simulado e apenas registra o payload final; nenhuma chamada de
rede é feita. Tokens usam o contador estimado da Sinapse e são rotulados como
simulados.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages
from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    build_request_body,
    load_deepseek_settings,
)
from .project_copilot import (
    FAROL_CANONICAL_PROMPT,
    ProjectStore,
    chat_with_deepseek,
)
from .software_context import classify_request

CREDENTIAL = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
OTHER_MARKER = "MARCADOR-OUTRO-123"
CONDITIONS = ("current", "recent_only", "selective", "semantic")


class CaptureClient:
    """Cliente de chat simulado que guarda o payload final."""

    simulated = True
    name = "audit-capture"

    def __init__(self, timings: dict[str, float] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.timings = timings

    def complete(self, request_body, timeout: float = 60.0):
        start = time.perf_counter()
        self.requests.append(copy.deepcopy(dict(request_body)))
        response = ChatCompletionResponse(
            content="resposta simulada",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={},
        )
        if self.timings is not None:
            self.timings["client_ms"] = round(
                (time.perf_counter() - start) * 1000, 2
            )
        return response


AUDIT_MESSAGES: tuple[tuple[str, str], ...] = (
    ("Objetivo: exportar a faixa final do projeto Aurora.", "user"),
    ("Decidi trocar o preset para Serum 2 porque a licença já estava comprada.", "user"),
    ("Certo — decisão vigente Serum 2.", "assistant"),
    ("Na verdade, troquei para Pigments porque a CPU do Aurora ficava alta.", "user"),
    ("O teste do refrão ficou concluído.", "user"),
    (
        "Se a CPU ficar abaixo de 30%, exportar em 48 kHz; senão, reduzir o oversampling.",
        "user",
    ),
    ("Ainda falta testar a ponte em 96 kHz.", "user"),
    ("Pensando em testar o preset Norte no baixo.", "user"),
    ("Também testei o preset Norte II no baixo e o resultado foi estável.", "user"),
    ("O preset Norte II usa dois envelopes e foi aprovado no baixo.", "user"),
    ("A CPU do Aurora ficou em 27%.", "user"),
    ("27% fica abaixo de 30%; o caminho é exportar.", "assistant"),
    ("Mudei de assunto: quero organizar os nomes das faixas.", "user"),
    ("A CPU ficou em 30%.", "user"),
    ("Qual é o nome do arquivo final e por que escolhi o Pigments?", "user"),
    ("Oi! Tudo bem? Como você está?", "user"),
    ("Estou pensando em tentar o preset Halo no pad.", "user"),
    ("As configurações de exportação são 48 kHz, 24 bits, dither ligado.", "user"),
    ("Para reduzir o oversampling, use 2x no render.", "user"),
    ("Você lembra do preset Norte II?", "user"),
)

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "casual",
        "project": "audit",
        "query": "Oi! Tudo bem? Como você está?",
        "expected": [],
    },
    {
        "id": "idea",
        "project": "audit",
        "query": "Estou pensando em tentar o preset Halo no pad.",
        "expected": [],
    },
    {
        "id": "resume",
        "project": "audit",
        "query": "Onde paramos?",
        "expected": ["m4", "m5", "m7"],
    },
    {
        "id": "corrected_decision",
        "project": "audit",
        "query": "Qual é a decisão vigente e por que?",
        "expected": ["m4"],
    },
    {
        "id": "goal",
        "project": "audit",
        "query": "Qual é o objetivo do projeto?",
        "expected": ["m1"],
    },
    {
        "id": "two_part",
        "project": "audit",
        "query": "Qual é o nome do arquivo final e por que escolhi o Pigments?",
        "expected": ["m4"],
    },
    {
        "id": "reason_without_source",
        "project": "reason",
        "query": "Por que escolhi o preset Norte para o arpejo?",
        "expected": ["m1"],
    },
    {
        "id": "old_similar_vocabulary",
        "project": "audit",
        "query": "O que decidimos sobre o preset Norte?",
        "expected": ["m8", "m9", "m10", "m20"],
    },
    {
        "id": "cpu_27",
        "project": "audit",
        "query": "A CPU do Aurora ficou em 27%.",
        "expected": ["m6", "m11"],
    },
    {
        "id": "cpu_30",
        "project": "audit",
        "query": "A CPU ficou em 30%.",
        "expected": ["m6", "m14"],
    },
    {
        "id": "budget_pressure",
        "project": "audit",
        "query": "Onde paramos?",
        "expected": ["m4", "m5", "m7"],
        "context_window": 260,
        "output_reserve": 80,
    },
    {
        "id": "norte_vs_norte_ii",
        "project": "audit",
        "query": "Compare o preset Norte com o Norte II no baixo.",
        "expected": ["m8", "m9", "m10", "m20"],
    },
    {
        "id": "topic_change",
        "project": "audit",
        "query": "Mudei de assunto: quero organizar os nomes das faixas.",
        "expected": [],
    },
    {
        "id": "delayed_result",
        "project": "audit",
        "query": "Como exporto em 48 kHz?",
        "expected": ["m6", "m18", "m19"],
        "reserved_ids": ["m18", "m19"],
    },
    {
        "id": "other_project",
        "project": "other",
        "query": "Onde paramos?",
        "expected": ["m1", "m2"],
    },
    {
        "id": "credentials",
        "project": "secret",
        "query": "Qual é a chave da API?",
        "expected": [],
    },
    {
        "id": "corrected_login",
        "project": "login",
        "query": "Qual é a decisão vigente sobre o login e por que?",
        "expected": ["m2"],
    },
    {
        "id": "different_component",
        "project": "login",
        "query": "O que decidimos sobre o cache?",
        "expected": [],
    },
    {
        "id": "historical_vocabulary",
        "project": "login",
        "query": "Por que tiramos o token do navegador?",
        "expected": ["m2"],
    },
    {
        "id": "linked_reason",
        "project": "login",
        "query": "Por que a sessão opaca foi escolhida?",
        "expected": ["m2"],
    },
    {
        "id": "what_changed",
        "project": "login",
        "query": "O que mudou na decisão de login?",
        "expected": ["m1", "m2"],
    },
    {
        "id": "two_part_partial",
        "project": "login",
        "query": "Qual é o teste pendente e por que tiramos o token do navegador?",
        "expected": ["m2", "m4"],
    },
    {
        "id": "unavailable_source",
        "project": "login",
        "query": "Qual é o motivo da mudança no cache?",
        "expected": [],
        "expected_gaps": ["decisão/motivo"],
    },
    {
        "id": "semantic_vocabulary",
        "project": "semantic",
        "query": "Por que abandonamos o cookie no cliente?",
        "expected": ["m2"],
    },
    {
        "id": "semantic_revoked",
        "project": "semantic",
        "query": "Por que trocamos o JWT no cliente?",
        "expected": ["m2"],
    },
    {
        "id": "semantic_unavailable",
        "project": "semantic",
        "query": "Por que removemos a fila de eventos?",
        "expected": [],
        "expected_gaps": ["decisão/motivo"],
    },
)


def _new_project(store: ProjectStore, name: str, messages: Sequence[tuple[str, str]]):
    project = store.create(name)
    for text, author in messages:
        store.add_message(project, text, author=author)
    return store.load(project["id"])


def build_fixture(root: str | Path) -> dict[str, Any]:
    from .integration import ProjectMemory

    store = ProjectStore(root)
    audit = _new_project(store, "Auditoria", AUDIT_MESSAGES)
    audit["project_goal"] = "exportar a faixa final do projeto Aurora"
    store.save(audit)
    reason = _new_project(
        store,
        "Sem motivo",
        (("Escolhi o preset Norte para o arpejo.", "user"),),
    )
    other = _new_project(
        store,
        "Outro projeto",
        (
            (f"{OTHER_MARKER} usa o preset Lume.", "user"),
            ("O próximo passo do outro projeto é revisar o arranjo.", "user"),
        ),
    )
    secret = _new_project(
        store,
        "Credencial",
        (("A chave fictícia de teste é sk-auditoria-ficticia-123456.", "user"),),
    )
    login = store.create("Login")
    login_repo = Path(root) / "repo-login"
    (login_repo / ".git").mkdir(parents=True, exist_ok=True)
    memory = ProjectMemory(
        login_repo, store=store, project_id=str(login["id"]), environ={}
    )
    memory.record(
        "architecture",
        "Usar JWT no login.",
        reason="API stateless.",
        files=["src/auth.py"],
        restrictions=["sem cookies"],
        entry_id="arch-login-jwt",
    )
    memory.record(
        "correction",
        "Usar sessão opaca no login.",
        supersedes="arch-login-jwt",
        reason="precisamos revogar tokens no navegador",
        files=["src/auth.py", "src/session.py"],
        restrictions=["sem JWT"],
        keywords=["token", "navegador", "cookie", "sessao", "servidor"],
        entry_id="arch-login-opaca",
    )
    memory.record(
        "test",
        "Teste do login com sessão opaca passou.",
        command="pytest tests/test_login.py",
        reported_by="opencode",
        report_source="relatório ses_login",
        entry_id="test-login-opaca",
    )
    memory.record(
        "pending",
        "Validar expiração da sessão.",
        command="pytest tests/test_session.py",
        entry_id="pend-login-exp",
    )
    for text in (
        "O time revisou os textos de ajuda do painel.",
        "Atualizamos o README do módulo de relatórios.",
        "O layout da tela inicial mudou nesta semana.",
        "Limpamos o cache local da máquina de testes.",
    ):
        store.add_message(store.load(login["id"]), text, author="user")

    semantic_project = store.create("Semântico")
    semantic_repo = Path(root) / "repo-semantico"
    (semantic_repo / ".git").mkdir(parents=True, exist_ok=True)
    semantic_memory = ProjectMemory(
        semantic_repo,
        store=store,
        project_id=str(semantic_project["id"]),
        environ={},
    )
    semantic_memory.record(
        "architecture",
        "Adotamos JWT no cliente.",
        reason="o time já conhecia o formato.",
        files=["src/auth.ts"],
        entry_id="sem-jwt",
    )
    semantic_memory.record(
        "correction",
        "Adotamos sessões opacas no servidor.",
        supersedes="sem-jwt",
        reason="precisamos revogar tokens no navegador.",
        files=["src/session.ts"],
        entry_id="sem-opaca",
    )
    semantic_memory.record(
        "test",
        "Teste de expiração da sessão passou.",
        command="pytest tests/test_session.py",
        reported_by="opencode",
        report_source="relatório sem_123",
        entry_id="sem-test",
    )
    semantic_memory.record(
        "pending",
        "Validar revogação no logout.",
        command="pytest tests/test_logout.py",
        entry_id="sem-pend",
    )
    for text in (
        "Revisamos os textos de ajuda.",
        "O layout da tela inicial mudou.",
        "Atualizamos a documentação do módulo.",
        "Limpamos o cache da máquina de testes.",
    ):
        store.add_message(store.load(semantic_project["id"]), text, author="user")
    return {
        "store": store,
        "projects": {
            "audit": audit["id"],
            "reason": reason["id"],
            "other": other["id"],
            "secret": secret["id"],
            "login": login["id"],
            "semantic": semantic_project["id"],
        },
    }


def _needs_retrieval(query: str) -> bool:
    return classify_request(query) != "casual"


def _extract_ids(text: str) -> set[str]:
    ids = set(re.findall(r"\b(m\d+)\b", text))
    ids.update(re.findall(r"id=(m\d+)", text))
    return ids


def _memory_section(text: str) -> str:
    match = re.search(
        r"=== MEMÓRIA \(Sinapse\) ===(.*?)=== FIM DA MEMÓRIA ===",
        text,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def _state_section(text: str) -> str:
    marker = "[ESTADO PERSISTIDO"
    index = text.find(marker)
    return text[index:] if index >= 0 else ""


def _analyze(
    body: Mapping[str, Any],
    query: str,
    expected: Sequence[str],
    current_id: str,
    status: str,
    timings: Mapping[str, float],
    project_key: str = "",
) -> dict[str, Any]:
    messages = [dict(message) for message in body.get("messages", ())]
    text = "\n".join(str(message.get("content", "")) for message in messages)
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    memory_text = _memory_section(system_text)
    state_text = _state_section(system_text)
    non_system = [
        message for message in messages if message.get("role") != "system"
    ]
    recent_text = "\n".join(
        str(message.get("content", "")) for message in non_system
    )
    sent = _extract_ids(text)
    memory_ids = sorted(_extract_ids(memory_text))
    state_ids = sorted(_extract_ids(state_text))
    recent_ids = sorted(_extract_ids(recent_text))
    expected_set = set(expected)
    missing = sorted(expected_set - sent)
    injected = set(memory_ids) | set(state_ids)
    irrelevant = sorted((injected - expected_set) - {current_id})
    query_in_recent = bool(
        len(non_system) >= 2
        and str(non_system[-2].get("content", "")).strip().endswith(query)
    )
    leaks = []
    if CREDENTIAL.search(text):
        leaks.append("credential")
    for marker in ("pred_", "reserved_ids", "search_query", "elapsed_ms", "hypothesis"):
        if marker in text:
            leaks.append(f"anticipation:{marker}")
    if project_key != "other" and OTHER_MARKER in text:
        leaks.append("other_project")
    if "[farol]" in text or "counter_name" in text or "within_radius" in text:
        leaks.append("technical_log")
    gaps = []
    for match in re.findall(r"Lacunas: ([^\n]+)", text):
        gaps.extend(part.strip() for part in match.split(";") if part.strip())
    tokens = EstimatedChatCounter(SimulatedCounter()).count_messages(messages)
    safe_messages = [
        {
            **message,
            "content": CREDENTIAL.sub("[credencial fictícia]", str(message.get("content", ""))),
        }
        for message in messages
    ]
    return {
        "status": status,
        "error": body.get("__error__"),
        "gaps": gaps,
        "sent_ids": sorted(sent),
        "memory_ids": memory_ids,
        "state_ids": state_ids,
        "recent_ids": recent_ids,
        "missing": missing,
        "irrelevant": irrelevant,
        "coverage": (
            round(len(expected_set & sent) / len(expected_set), 3)
            if expected_set
            else 1.0
        ),
        "tokens": tokens,
        "memory_tokens": (
            SimulatedCounter().count(memory_text) if memory_text else 0
        ),
        "duplicate_query": query_in_recent,
        "leaks": leaks,
        "latency_ms": {key: round(float(value), 2) for key, value in timings.items()},
        "messages": safe_messages,
    }


def _timed(fn: Callable, timings: dict[str, float], name: str):
    start = time.perf_counter()
    try:
        return fn()
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 2)


def _run_recent_only(
    project: Mapping[str, Any],
    query: str,
    config: Mapping[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    previous = list(project.get("messages", ()))[:-1][-4:]
    history = [
        RecentTurn(text=str(message["text"]), id=str(message["id"]), author=str(message["author"]))
        for message in previous
    ]
    settings = load_deepseek_settings(config)
    messages = _timed(
        lambda: build_base_messages(FAROL_CANONICAL_PROMPT, history, query),
        timings,
        "build_ms",
    )
    body = _timed(
        lambda: build_request_body(messages, settings),
        timings,
        "request_ms",
    )
    return dict(body)


def _run_current(
    project: Mapping[str, Any],
    query: str,
    config: Mapping[str, Any],
    store: ProjectStore,
    timings: dict[str, float],
    *,
    reserved_ids: Sequence[str] = (),
    semantic: bool = False,
) -> tuple[dict[str, Any], str]:
    import sinapse.chatbot as chatbot_module

    config = copy.deepcopy(dict(config))
    config.setdefault("semantic", {})["enabled"] = semantic

    original_recall = chatbot_module.recall
    original_select = chatbot_module.select_chat_context
    budget_status = "ok"

    def timed_recall(*args, **kwargs):
        start = time.perf_counter()
        result = original_recall(*args, **kwargs)
        timings["recall_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    def timed_select(*args, **kwargs):
        nonlocal budget_status
        start = time.perf_counter()
        result = original_select(*args, **kwargs)
        timings["select_ms"] = round((time.perf_counter() - start) * 1000, 2)
        budget_status = result.diagnostics.status
        return result

    client = CaptureClient(timings)
    chatbot_module.recall = timed_recall
    chatbot_module.select_chat_context = timed_select
    start = time.perf_counter()
    try:
        chat_with_deepseek(
            project,
            query,
            config,
            reserved_ids=reserved_ids,
            store=store,
            client=client,
            model="deepseek-flash",
        )
    except ValueError as error:
        timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return {"messages": [], "__error__": str(error)}, "error"
    finally:
        chatbot_module.recall = original_recall
        chatbot_module.select_chat_context = original_select
        if "total_ms" not in timings:
            timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
    timings["prepare_ms"] = round(
        timings["total_ms"] - timings.get("client_ms", 0.0), 2
    )
    return client.requests[-1], budget_status


def run_audit(
    root: str | Path | None = None,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temporary = None
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="sinapse-audit-")
        root = temporary.name
    base_config = copy.deepcopy(dict(config or load_config()))
    fixture = build_fixture(root)
    store: ProjectStore = fixture["store"]
    results = []
    for case in cases or CASES:
        project_id = fixture["projects"][case["project"]]
        project = copy.deepcopy(store.load(project_id))
        query = str(case["query"])
        current_id = f"m{len(project['messages']) + 1}"
        project["messages"].append(
            {
                "id": current_id,
                "text": query,
                "author": "user",
                "created_at": "auditoria",
            }
        )
        case_config = copy.deepcopy(base_config)
        if case.get("context_window"):
            case_config.setdefault("deepseek", {})
            case_config["deepseek"]["context_window"] = int(case["context_window"])
            case_config["deepseek"]["output_reserve"] = int(
                case.get("output_reserve", 80)
            )
            case_config["deepseek"]["max_tokens"] = int(
                case.get("output_reserve", 80)
            )
        if case.get("reserved_ids"):
            project["anticipatory"] = {
                "status": "ready",
                "hypothesis": "pred_AUDIT",
                "source_ids": list(case["reserved_ids"]),
                "turn": project.get("turn", 0),
                "project_revision": project.get("revision", 0),
            }
        before = json.dumps(project.get("state", {}), sort_keys=True, ensure_ascii=False)
        entry = {
            "id": case["id"],
            "project": case["project"],
            "query": query,
            "current_id": current_id,
            "expected": list(case["expected"]),
            "expected_gaps": list(case.get("expected_gaps", ())),
            "conditions": {},
        }
        for condition in CONDITIONS:
            timings: dict[str, float] = {}
            if condition == "recent_only" or (
                condition == "selective" and not _needs_retrieval(query)
            ):
                body = _run_recent_only(project, query, case_config, timings)
                status = "recent_only"
            else:
                body, status = _run_current(
                    project,
                    query,
                    case_config,
                    store,
                    timings,
                    reserved_ids=case.get("reserved_ids", ()),
                    semantic=condition == "semantic",
                )
            entry["conditions"][condition] = _analyze(
                body,
                query,
                case["expected"],
                current_id,
                status,
                timings,
                case["project"],
            )
        after = json.dumps(project.get("state", {}), sort_keys=True, ensure_ascii=False)
        entry["state_changed"] = before != after
        results.append(entry)
    if temporary is not None:
        temporary.cleanup()
    summary = {}
    for condition in CONDITIONS:
        items = [entry["conditions"][condition] for entry in results]
        required = [
            entry["conditions"][condition]
            for entry in results
            if entry["expected"]
        ]
        summary[condition] = {
            "coverage_avg": round(
                sum(item["coverage"] for item in items) / len(items), 3
            ),
            "coverage_required_avg": round(
                sum(item["coverage"] for item in required) / len(required), 3
            )
            if required
            else 1.0,
            "missing_total": sum(len(item["missing"]) for item in items),
            "irrelevant_total": sum(len(item["irrelevant"]) for item in items),
            "avg_tokens": round(sum(item["tokens"] for item in items) / len(items), 1),
            "duplicate_cases": sum(1 for item in items if item["duplicate_query"]),
            "leak_cases": sum(1 for item in items if item["leaks"]),
            "base_overflow_cases": sum(
                1 for item in items if item["status"] == "base_overflow"
            ),
            "avg_prepare_ms": round(
                sum(
                    item["latency_ms"].get("prepare_ms", item["latency_ms"].get("build_ms", 0.0))
                    for item in items
                )
                / len(items),
                2,
            ),
        }
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "deepseek-flash",
        "client": "simulated",
        "counter": {"name": "simulated-words-punct-v1", "simulated": True},
        "cases": results,
        "summary": summary,
        "state_changes": sum(1 for entry in results if entry["state_changed"]),
        "limitations": [
            "Cliente simulado: citações do modelo não são observáveis.",
            "Sem DEEPSEEK_API_KEY nem chave no painel: nenhuma chamada real.",
            "Tokens são estimativa do contador da Sinapse, não do tokenizador oficial.",
        ],
    }


def format_report(result: Mapping[str, Any]) -> str:
    lines = [
        "AUDITORIA DE CONTEXTO — FAROL",
        f"Modelo: {result['model']} | cliente: {result['client']} | "
        f"estado alterado: {result['state_changes']}",
        "",
        f"{'caso':<24} {'cond':<12} {'cob':>4} {'falta':>5} {'irrel':>5} "
        f"{'tokens':>6} {'dup':>3} {'leak':>4}",
    ]
    for entry in result["cases"]:
        for condition in CONDITIONS:
            item = entry["conditions"][condition]
            lines.append(
                f"{entry['id']:<24} {condition:<12} {item['coverage']:>4.2f} "
                f"{len(item['missing']):>5} {len(item['irrelevant']):>5} "
                f"{item['tokens']:>6} {'sim' if item['duplicate_query'] else '-':>3} "
                f"{len(item['leaks']):>4}"
            )
    lines.append("")
    for condition, item in result["summary"].items():
        lines.append(
            f"{condition:<12} cobertura={item['coverage_avg']:.2f} "
            f"(exigida={item['coverage_required_avg']:.2f}) "
            f"faltas={item['missing_total']} irrel={item['irrelevant_total']} "
            f"tokens={item['avg_tokens']} duplicados={item['duplicate_cases']} "
            f"leaks={item['leak_cases']} overflow={item['base_overflow_cases']} "
            f"preparo={item['avg_prepare_ms']}ms"
        )
    return "\n".join(lines)
