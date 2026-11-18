"""Seleção de contexto técnico do Farol por tipo de pedido e assunto."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .budget import SimulatedCounter
from .semantic import SemanticIndex
from .sensitivity import SensitivityPolicy
from .text import normalize_text, tokenize

GENERIC = {
    "o", "a", "os", "as", "de", "da", "do", "das", "dos", "e", "que", "qual",
    "quais", "como", "por", "porque", "para", "pra", "em", "no", "na", "um",
    "uma", "me", "sobre", "tem", "foi", "ser", "esta", "está", "isso", "aqui",
    "agora", "ainda", "mais", "menos", "favor", "pode", "poderia", "quero",
    "preciso", "ver", "mostrar", "mostra", "conta", "dizer", "saber", "onde",
    "paramos", "retomar", "tarefa", "atual", "voce", "você",
    "decidimos", "decisao", "decisão", "tecnica", "técnica", "arquitetura",
    "bug", "erro", "falha", "resultado", "teste", "test", "componente",
    "diferente", "outro", "outra", "coisa", "sobre", "fazer", "feito",
    "vigente", "ficou", "ficado", "ficar", "esta", "está", "estava", "deu",
    "passou", "aconteceu", "mediu", "medido", "valor", "usando", "usa",
    "usar", "precisa", "precisamos", "queria", "queremos", "vamos", "vou",
    "mudou", "mudanca", "mudança", "corrig", "correção", "correcao", "antes",
    "historico", "histórico",
}

CASUAL = (
    "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "tudo bem",
    "como voce esta", "como você está", "obrigado", "obrigada", "valeu",
    "ate mais", "até mais", "bom trabalho",
)
IMPLEMENT = (
    "instrucoes", "instruções", "instrucao", "instrução", "implementar",
    "implemente", "corrigir", "corrija", "passar para o opencode",
    "briefing", "prompt para", "plano de acao", "plano de ação",
    "tarefa para o opencode",
)
BUG = (
    "bug", "erro", "falha", "quebrou", "nao funciona", "não funciona",
    "crash", "traceback", "exception", "regressao", "regressão", "defeito",
)
ARCHITECTURE = (
    "arquitetura", "decisao tecnica", "decisão técnica", "decisao vigente",
    "decisão vigente", "adr", "trade-off", "tradeoff", "design",
    "por que escolhemos", "por que escolheu", "por que escolhi",
    "decisao de arquitetura", "decisão de arquitetura", "decidimos sobre",
)
RESUME = ("onde paramos", "retomar", "como estamos", "proximo passo", "próximo passo")
TEST = ("teste", "test", "rodou", "resultado do teste", "suite", "cobertura", "custo", "custou")
RESULT = ("cpu", "resultado", "medi", "mediu", "porcent", "%", "27", "30")
GOAL = ("objetivo", "meta do projeto", "goal")
IDEA = ("pensando", "talvez", "ideia", "e se", "considerando", "sera que")
TECHNICAL_GENERAL = (
    "export", "exportar", "exporto", "configura", "configuracao",
    "configuração", "refator", "migrar", "deploy", "login", "sessao",
    "sessão", "autenticacao", "autenticação", "compare", "comparar",
    "comparacao", "comparação", "diferenca", "diferença",
)
DOMAIN_TERMS = (
    "farol", "sinapse", "opencode", "open code", "memoria", "memória",
    "contexto", "busca", "semantica", "semântica", "chave", "token",
    "navegador", "projeto", "teste", "cobertura", "duplic", "credencial",
    "fonte", "lacuna", "repositorio", "repositório",
)


def _has_cue(normalized: str, cues: Sequence[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(cue)}\b", normalized) for cue in cues
    )


def classify_request(text: str) -> str:
    return str(analyze_request(text)["kind"])


def analyze_request(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        return {"kind": "general", "intents": [], "subject": []}
    tokens = set(normalized.split())
    technical = _has_cue(
        normalized, IMPLEMENT + BUG + ARCHITECTURE + RESUME + TEST + RESULT + GOAL
        + TECHNICAL_GENERAL
    )
    if not technical and (
        _has_cue(normalized, CASUAL)
        or "mudei de assunto" in normalized
        or (len(tokens) <= 4 and _has_cue(normalized, CASUAL))
    ):
        return {"kind": "casual", "intents": [], "subject": []}

    intents: list[str] = []
    if _has_cue(normalized, IMPLEMENT):
        intents.append("implementation")
    if _has_cue(normalized, BUG):
        intents.append("bug")
    if _has_cue(normalized, ARCHITECTURE) or any(
        cue in normalized for cue in ("motivo", "por que", "porque")
    ):
        intents.append("decision")
    if _has_cue(normalized, RESUME):
        intents.append("resume")
    if _has_cue(normalized, RESULT) and (
        re.search(r"\b\d{1,3}\s*%", normalized) or "cpu" in normalized
        or "resultado" in normalized
    ):
        intents.append("result_condition")
    if any(
        cue in normalized
        for cue in ("falta", "faltando", "pendente", "pendencia", "pendência")
    ):
        intents.append("pending_test")
    test_cue = _has_cue(normalized, TEST)
    if test_cue:
        if any(
            cue in normalized
            for cue in ("falta", "pendente", "pendencia", "pendência", "qual teste", "quais testes")
        ):
            intents.append("pending_test")
        else:
            intents.append("test")
    if _has_cue(normalized, GOAL) or "objetivo" in normalized:
        intents.append("goal")
    if any(
        cue in normalized
        for cue in ("mudou", "corrig", "correção", "correcao", "antes", "histórico", "historico", "na verdade")
    ):
        intents.append("correction_history")

    domain = any(term in normalized for term in DOMAIN_TERMS)
    if domain:
        intents.append("technical")
        if any(
            cue in normalized
            for cue in ("por que", "porque", "como", "qual", "quais")
        ) and "implementation" not in intents:
            intents.append("decision")

    if "implementation" in intents:
        kind = "implement"
    elif "bug" in intents:
        kind = "bug"
    elif "decision" in intents:
        kind = "architecture"
    elif "resume" in intents:
        kind = "resume"
    elif "result_condition" in intents:
        kind = "result"
    elif "test" in intents or "pending_test" in intents:
        kind = "test"
    elif "goal" in intents:
        kind = "goal"
    elif _has_cue(normalized, TECHNICAL_GENERAL):
        kind = "technical"
    else:
        kind = "general"
    return {
        "kind": kind,
        "intents": list(dict.fromkeys(intents)) or [kind],
        "subject": sorted(subject_tokens(text)),
    }


def subject_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in GENERIC}


def _policy(project: Mapping[str, Any]) -> SensitivityPolicy:
    policy = SensitivityPolicy()
    for restriction in project.get("restrictions", ()):
        policy.declare(str(project["id"]), str(restriction), 0)
    return policy


def _visible(policy: SensitivityPolicy, project: Mapping[str, Any], text: str, query: str) -> bool:
    return not policy.text_is_blocked(str(project["id"]), str(text), query)[0]


def _text(entry: Mapping[str, Any]) -> str:
    return str(
        entry.get("value")
        or entry.get("text")
        or entry.get("to")
        or entry.get("from")
        or entry.get("symptom")
        or ""
    )


def _entry_tokens(entry: Mapping[str, Any]) -> set[str]:
    parts = [
        _text(entry),
        str(entry.get("reason") or ""),
        str(entry.get("from") or ""),
        str(entry.get("to") or ""),
        str(entry.get("component") or ""),
        " ".join(entry.get("files", ()) or ()),
        " ".join(entry.get("restrictions", ()) or ()),
        " ".join(entry.get("keywords", ()) or ()),
        str(entry.get("symptom") or ""),
        str(entry.get("command") or ""),
    ]
    return set(tokenize(" ".join(parts)))


def _matches(entry: Mapping[str, Any], subject: set[str]) -> bool:
    return not subject or bool(_entry_tokens(entry) & subject)


def _source_ids(entry: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in entry.get("source_ids", ())]


def _refs(entry: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "files", "issues", "prs", "commits", "command", "report_source",
        "reported_by", "restrictions", "alternatives", "symptom", "status",
        "entry_id", "corrects_entry_id", "correction_source_ids",
    )
    return {key: entry[key] for key in keys if entry.get(key)}


def _message_entry(message: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": message.get("text", ""),
        "source_ids": [str(message.get("id", ""))],
        "kind": message.get("kind", "note"),
        "entry_id": message.get("entry_id"),
        **{key: message[key] for key in (
            "files", "issues", "prs", "commits", "command", "report_source",
            "reported_by", "restrictions", "alternatives", "symptom",
            "status", "related_to",
        ) if message.get(key)},
    }


def _results(project: Mapping[str, Any], subject: set[str], policy, query) -> list[dict]:
    current_id = None
    for message in reversed(project.get("messages", ())):
        if (
            message.get("author") == "user"
            and str(message.get("text", "")).strip() == str(query).strip()
        ):
            current_id = str(message.get("id", ""))
            break
    scored = []
    for message in project.get("messages", ()):
        if message.get("author") == "assistant" and not message.get("kind"):
            continue
        text = str(message.get("text", ""))
        if current_id and str(message.get("id", "")) == current_id:
            continue
        if not re.search(r"\b\d{1,3}\s*%", text):
            continue
        normalized = normalize_text(text)
        if re.search(r"\bse\b", normalized) and not re.search(
            r"\b(ficou|resultado|mediu|medido)\b", normalized
        ):
            continue
        if not _visible(policy, project, text, query):
            continue
        entry = _message_entry(message)
        overlap = len(_entry_tokens(entry) & subject)
        if not subject or overlap:
            scored.append((overlap, int(str(message.get("id", "m0"))[1:] or 0), entry))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [entry for _, _, entry in scored[-1:]]


def _conditions(project: Mapping[str, Any], subject: set[str], policy, query) -> list[dict]:
    found = []
    for message in project.get("messages", ()):
        text = str(message.get("text", ""))
        normalized = normalize_text(text)
        if not re.search(r"\bse\b.*(abaixo|acima|maior|menor|entao|então)", normalized):
            if not re.search(r"\bquando\b", normalized):
                continue
        if not _visible(policy, project, text, query):
            continue
        entry = _message_entry(message)
        if _matches(entry, subject) or not subject:
            found.append(entry)
    return found[-1:]


def _linked_entries(
    project: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    policy: SensitivityPolicy,
    query: str,
) -> list[dict]:
    """Segue vínculos explícitos antes de qualquer busca lexical."""
    by_id = {str(message.get("id")): message for message in project.get("messages", ())}
    selected_ids = {
        source for entry in selected for source in _source_ids(entry)
    }
    linked: list[dict] = []
    seen: set[str] = set()
    for entry in selected:
        ids: list[str] = []
        for key in (
            "reason_source_ids",
            "correction_source_ids",
            "corrects_source_ids",
            "resolved_by",
            "completed_by",
            "related_to",
        ):
            value = entry.get(key)
            if isinstance(value, str):
                ids.append(value)
            elif value:
                ids.extend(str(item) for item in value)
        for source_id in ids:
            if source_id in selected_ids or source_id in seen:
                continue
            message = by_id.get(source_id)
            if message is None:
                continue
            if not _visible(policy, project, str(message.get("text", "")), query):
                continue
            seen.add(source_id)
            linked.append({**_message_entry(message), "linked": True})
    return linked


def _related_messages(
    project: Mapping[str, Any], subject: set[str], policy, query, limit: int = 6
) -> list[dict]:
    if not subject:
        return []
    scored = []
    for message in project.get("messages", ()):
        if message.get("author") == "assistant" and not message.get("kind"):
            continue
        text = str(message.get("text", ""))
        if normalize_text(text) == normalize_text(query):
            continue
        if not _visible(policy, project, text, query):
            continue
        tokens = set(tokenize(text))
        overlap = len(subject & tokens)
        if overlap:
            scored.append((overlap, str(message.get("id", "")), _message_entry(message)))
    if not scored:
        return []
    threshold = 2
    scored = [item for item in scored if item[0] >= threshold]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry for _, _, entry in scored[:limit]]


def _goal(project: Mapping[str, Any]) -> dict[str, Any] | None:
    goal = project.get("project_goal")
    if not goal:
        return None
    source_ids = [
        str(message["id"])
        for message in project.get("messages", ())
        if message.get("kind") == "goal"
        or (
            message.get("author") == "user"
            and "objetivo" in normalize_text(str(message.get("text", "")))
        )
    ]
    return {"text": str(goal), "source_ids": source_ids[:1]}


def _task(project: Mapping[str, Any]) -> dict[str, Any] | None:
    task = project.get("current_task")
    if task:
        source_ids = [
            str(message["id"])
            for message in project.get("messages", ())
            if message.get("kind") == "task"
            or (
                message.get("author") == "user"
                and str(message.get("text", "")).strip() == str(task).strip()
            )
        ]
        return {"text": str(task), "source_ids": source_ids[-1:] if source_ids else []}
    user_messages = [
        message for message in project.get("messages", ()) if message.get("author") == "user"
    ]
    if user_messages:
        return _message_entry(user_messages[-1])
    return None


def select_state(
    project: Mapping[str, Any],
    query: str,
    kind: str | None = None,
    *,
    semantic: bool = False,
) -> dict[str, Any]:
    plan = analyze_request(query)
    kind = kind or str(plan["kind"])
    intents = set(plan["intents"])
    if kind not in intents:
        intents.add(kind)
    subject = subject_tokens(query)
    policy = _policy(project)
    state = project.get("state", {})

    wants_decision = bool(intents & {"decision", "architecture", "resume", "bug", "implement", "implementation", "correction_history"})
    wants_goal = bool(intents & {"goal", "implement", "implementation"})
    wants_test = bool(intents & {"pending_test", "test", "resume", "bug", "implement", "implementation"})
    wants_condition = bool(intents & {"result_condition", "bug", "test"}) or "export" in normalize_text(query)
    wants_correction = "correction_history" in intents

    decisions = [
        item for item in state.get("decisions", ())
        if item.get("status") == "vigente"
        and _visible(policy, project, _text(item), query)
    ]
    matching = [item for item in decisions if _matches(item, subject)]
    if subject:
        scored = [
            (
                len(_entry_tokens(item) & subject)
                + (2 if item.get("component") in subject else 0),
                index,
                item,
            )
            for index, item in enumerate(matching)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        decision = scored[-1][2] if scored and scored[-1][0] > 0 else None
    else:
        decision = decisions[-1] if decisions else None
    if not wants_decision or (
        kind in {"result", "technical"}
        and not any(cue in normalize_text(query) for cue in ("decisao", "decisão", "arquitetura", "escolha"))
    ):
        decision = None

    all_bugs = [
        item for item in state.get("bugs", ())
        if _visible(policy, project, _text(item), query)
    ]
    wants_bug = "bug" in intents or "implementation" in intents
    matching_bugs = [item for item in all_bugs if _matches(item, subject)]
    if wants_bug:
        bugs = matching_bugs if subject else all_bugs[-2:]
    elif "resume" in intents:
        bugs = [
            item for item in all_bugs
            if item.get("status") in {"aberto", "open"}
        ][-1:]
    else:
        bugs = []

    tests = [
        item for item in state.get("completed", ())
        if item.get("status", "verified") == "verified"
        and _visible(policy, project, _text(item), query)
    ]
    if subject:
        tests = [item for item in tests if _matches(item, subject)]
    if wants_test:
        tests = tests[-2:]
    else:
        tests = []

    pending = [
        item for item in state.get("pending", ())
        if item.get("status") == "aberto"
        and _visible(policy, project, _text(item), query)
    ]
    if subject and "bug" in intents:
        pending = [item for item in pending if _matches(item, subject)]
    else:
        pending = pending[-2:]
    if not wants_test:
        pending = []

    correction_query = any(
        cue in normalize_text(query)
        for cue in ("corrig", "correção", "correcao", "mudou", "na verdade", "histórico", "historico")
    )
    corrections = [
        item for item in project.get("corrections", ())
        if (not subject or _matches(item, subject)) and wants_correction
    ]
    changes = [
        item for item in state.get("changes", ())
        if not subject or _matches(item, subject)
    ]
    hypotheses = [
        item for item in state.get("hypotheses", ())
        if not subject or _matches(item, subject)
    ]
    conditions = (
        _conditions(project, subject, policy, query) if wants_condition else []
    )
    results = (
        _results(project, subject, policy, query)
        if wants_condition and "result_condition" in intents
        else []
    )

    selected_ids = {
        source
        for item in (
            ([decision] if decision else [])
            + ([_goal(project)] if (_goal(project)) else [])
            + bugs
            + tests
            + pending
            + conditions
            + results
        )
        for source in _source_ids(item)
    }
    decision_ids = {
        source
        for item in state.get("decisions", ())
        for source in _source_ids(item)
    }
    evidence = (
        ([decision] if decision else [])
        + bugs
        + tests
        + pending
        + conditions
        + results
    )
    covered_tokens = set()
    for item in evidence:
        covered_tokens.update(_entry_tokens(item))
    missing_subject = subject - covered_tokens
    linked = _linked_entries(
        project,
        evidence + (corrections if wants_correction else []),
        policy,
        query,
    )
    lexical = (
        _related_messages(project, missing_subject or subject, policy, query)
        if (missing_subject or wants_bug)
        else []
    )
    related = linked + lexical
    related = [
        item for item in related
        if not selected_ids.intersection(_source_ids(item))
        and (
            correction_query
            or not decision_ids.intersection(_source_ids(item))
        )
    ]
    related = list(
        {
            source: item
            for item in related
            for source in _source_ids(item)[:1]
        }.values()
    )

    checks = {
        "decision": bool(decision),
        "goal": bool(_goal(project)),
        "pending_test": bool(pending or tests),
        "result_condition": bool(conditions or results),
        "bug": bool(bugs),
        "correction_history": bool(corrections),
        "test": bool(tests),
    }
    requested = set(intents)
    if "resume" in requested:
        requested.update({"decision", "test", "pending_test"})
    labels = {
        "decision": "decisão/motivo",
        "goal": "objetivo",
        "pending_test": "teste pendente",
        "result_condition": "condição/resultado",
        "bug": "bug",
        "correction_history": "histórico de correção",
        "test": "teste",
    }
    coverage = [labels[key] for key in checks if key in requested and checks[key]]
    gaps = [labels[key] for key in checks if key in requested and not checks[key]]

    if semantic and gaps:
        hits = SemanticIndex(project).search(
            query, limit=6, min_score=0.16, exclude=selected_ids
        )
        if "decisão/motivo" in gaps and decision is None:
            for hit in hits:
                if hit["score"] < 0.22:
                    continue
                candidate = next(
                    (
                        item
                        for item in state.get("decisions", ())
                        if item.get("status") == "vigente"
                        and set(item.get("source_ids", ())) & set(hit["source_ids"])
                    ),
                    None,
                )
                if candidate is not None:
                    decision = candidate
                    break
        for hit in hits:
            if selected_ids.intersection(hit["source_ids"]):
                continue
            if decision and set(decision.get("source_ids", ())) & set(hit["source_ids"]):
                continue
            if not _visible(policy, project, hit.get("text", ""), query):
                continue
            if "bug" in gaps and hit["kind"] == "bug" and hit["score"] >= 0.22:
                bugs.append(hit["raw"])
                selected_ids.update(hit["source_ids"])
                continue
            if (
                "teste pendente" in gaps
                and hit["kind"] in {"pending", "test-suggested"}
                and hit["score"] >= 0.22
            ):
                pending.append(hit["raw"])
                selected_ids.update(hit["source_ids"])
                continue
            if "teste" in gaps and hit["kind"] == "test" and hit["score"] >= 0.18:
                tests.append(hit["raw"])
                selected_ids.update(hit["source_ids"])
                continue
        checks["decision"] = bool(decision)
        checks["bug"] = bool(bugs)
        checks["pending_test"] = bool(pending or tests)
        checks["test"] = bool(tests)
        coverage = [labels[key] for key in checks if key in requested and checks[key]]
        gaps = [labels[key] for key in checks if key in requested and not checks[key]]

    return {
        "kind": kind,
        "intents": sorted(requested),
        "subject": sorted(subject),
        "task": _task(project),
        "goal": _goal(project) if wants_goal or "objetivo" in normalize_text(query) else None,
        "decision": decision,
        "bugs": bugs[-2:],
        "conditions": conditions,
        "results": results,
        "tests": tests,
        "pending": pending[-2:],
        "corrections": corrections[-2:],
        "changes": changes[-2:],
        "hypotheses": hypotheses[-2:],
        "related": related,
        "coverage": coverage,
        "gaps": gaps,
        "restricted": [],
    }


def _line(entry: Mapping[str, Any], label: str) -> str:
    ids = _source_ids(entry)
    suffix = " [" + ", ".join(ids) + "]" if ids else ""
    return f"{label}: {_text(entry)}{suffix}"


def _details(entry: Mapping[str, Any]) -> list[str]:
    lines = []
    if entry.get("reason"):
        lines.append(f"  Motivo: {entry['reason']}")
    for key, label in (
        ("restrictions", "Restrições"),
        ("files", "Arquivos"),
        ("keywords", "Palavras-chave"),
        ("issues", "Issues"),
        ("prs", "PRs"),
        ("commits", "Commits"),
        ("command", "Comando"),
        ("report_source", "Relatório"),
        ("reported_by", "Origem"),
        ("symptom", "Sintoma"),
        ("alternatives", "Alternativas"),
    ):
        value = entry.get(key)
        if value:
            rendered = "; ".join(value) if isinstance(value, (list, tuple)) else str(value)
            lines.append(f"  {label}: {rendered}")
    return lines


def render_state(fields: Mapping[str, Any]) -> str:
    lines = ["[ESTADO PERSISTIDO - PROJETO]"]
    task = fields.get("task")
    if task:
        lines.append(_line(task, "Tarefa atual"))
    goal = fields.get("goal")
    if goal:
        lines.append(_line(goal, "Objetivo"))
    decision = fields.get("decision")
    if decision:
        lines.append(_line(decision, "Decisão técnica"))
        lines.extend(_details(decision))
    else:
        lines.append("Decisão técnica: nenhuma decisão vigente sobre este assunto")
    for correction in fields.get("corrections", ()):
        ids = _source_ids(correction) + [
            str(item) for item in correction.get("correction_source_ids", ())
        ]
        suffix = " [" + ", ".join(dict.fromkeys(ids)) + "]" if ids else ""
        lines.append(f"Correção: {correction.get('from')} → {correction.get('to')}{suffix}")
    for bug in fields.get("bugs", ()):
        label = "Bug aberto" if bug.get("status") in {"aberto", "open"} else "Bug corrigido"
        lines.append(_line(bug, label))
        lines.extend(_details(bug))
    for condition in fields.get("conditions", ()):
        lines.append(_line(condition, "Condição aplicável"))
    for result in fields.get("results", ()):
        lines.append(_line(result, "Resultado observado"))
    for test in fields.get("tests", ()):
        lines.append(_line(test, "Último teste observado"))
        lines.extend(_details(test))
    for change in fields.get("changes", ()):
        lines.append(_line(change, "Mudança"))
        lines.extend(_details(change))
    for hypothesis in fields.get("hypotheses", ()):
        lines.append(_line(hypothesis, "Hipótese"))
    for pending in fields.get("pending", ()):
        lines.append(_line(pending, "Pendência"))
        lines.extend(_details(pending))
    for related in fields.get("related", ()):
        lines.append(_line(related, "Evidência relacionada"))
        lines.extend(_details(related))
    if len(lines) == 1:
        lines.append("Nenhum estado técnico registrado para este pedido.")
    if fields.get("gaps"):
        lines.append("Lacunas: " + "; ".join(fields["gaps"]))
    return "\n".join(lines)


def build_context(
    project: Mapping[str, Any],
    query: str,
    kind: str | None = None,
    max_tokens: int = 260,
    *,
    semantic: bool = False,
) -> tuple[str, list[str], bool]:
    fields = select_state(project, query, kind, semantic=semantic)
    text = render_state(fields)
    ids = [
        source
        for entry in (
            [fields.get("task"), fields.get("goal")]
            + list(fields.get("bugs", ()))
            + list(fields.get("tests", ()))
            + list(fields.get("pending", ()))
            + list(fields.get("related", ()))
            + ([fields["decision"]] if fields.get("decision") else [])
        )
        if entry
        for source in _source_ids(entry)
    ]
    counter = SimulatedCounter()
    if counter.count(text) <= max_tokens:
        return text, list(dict.fromkeys(ids)), False
    lines = text.splitlines()
    kept = [lines[0]]
    truncated = False
    for line in lines[1:]:
        if counter.count("\n".join(kept + [line])) > max_tokens:
            truncated = True
            break
        kept.append(line)
    return "\n".join(kept), list(dict.fromkeys(ids)), truncated


def format_bug(project: Mapping[str, Any], query: str, *, semantic: bool = False) -> str:
    fields = select_state(project, query, "bug", semantic=semantic)
    bugs = fields.get("bugs") or []
    if not bugs:
        subject = ", ".join(fields.get("subject", ())) or query
        return (
            f"Não encontrei bug registrado sobre “{subject}”. "
            "Se você descrever o sintoma, eu registro com arquivos e teste."
        )
    lines = []
    for bug in bugs[-1:]:
        label = "Bug aberto" if bug.get("status") in {"aberto", "open"} else "Bug corrigido"
        lines.append(_line(bug, label))
        lines.extend(_details(bug))
        for related in fields.get("related", ()):
            if related.get("entry_id") == bug.get("entry_id") or not bug.get("entry_id"):
                lines.append(_line(related, "Tentativa/evidência"))
        for test in fields.get("tests", ()):
            lines.append(_line(test, "Teste relacionado"))
    pending = fields.get("pending") or []
    if pending:
        lines.append(_line(pending[-1], "Próximo passo"))
    if fields.get("gaps"):
        lines.append("Lacunas: " + "; ".join(fields["gaps"]))
    return "\n".join(lines)


def format_architecture(project: Mapping[str, Any], query: str, *, semantic: bool = False) -> str:
    fields = select_state(project, query, "architecture", semantic=semantic)
    decision = fields.get("decision")
    if not decision:
        return "Não há decisão técnica vigente registrada sobre esse componente."
    lines = [_line(decision, "Decisão técnica"), *_details(decision)]
    for correction in fields.get("corrections", ()):
        ids = _source_ids(correction) + [
            str(item) for item in correction.get("correction_source_ids", ())
        ]
        suffix = " [" + ", ".join(dict.fromkeys(ids)) + "]" if ids else ""
        lines.append(f"Correção: {correction.get('from')} → {correction.get('to')}{suffix}")
    for test in fields.get("tests", ()):
        lines.append(_line(test, "Teste observado"))
    for pending in fields.get("pending", ()):
        lines.append(_line(pending, "Teste/pendência"))
    if fields.get("gaps"):
        lines.append("Lacunas: " + "; ".join(fields["gaps"]))
    return "\n".join(lines)


def format_test(project: Mapping[str, Any], query: str, *, semantic: bool = False) -> str:
    fields = select_state(project, query, "test", semantic=semantic)
    tests = fields.get("tests") or []
    if not tests:
        return "Ainda não há teste observado registrado para esse assunto."
    test = tests[-1]
    lines = [_line(test, "Último teste observado"), *_details(test)]
    if fields.get("conditions"):
        lines.append(_line(fields["conditions"][-1], "Condição aplicável"))
    if fields.get("gaps"):
        lines.append("Lacunas: " + "; ".join(fields["gaps"]))
    return "\n".join(lines)


def format_brief(project: Mapping[str, Any], query: str, *, semantic: bool = False) -> str:
    fields = select_state(project, query, "implement", semantic=semantic)
    task = fields.get("task") or {}
    goal = fields.get("goal") or {}
    decision = fields.get("decision") or {}
    bug = (fields.get("bugs") or [None])[-1]
    tests = fields.get("tests") or []
    pending = fields.get("pending") or []

    lines = ["**Objetivo**"]
    objective = goal.get("text") or task.get("text") or query
    lines.append(f"- {objective}" + (
        " [" + ", ".join(_source_ids(goal or task)) + "]"
        if _source_ids(goal or task) else ""
    ))

    lines.append("")
    lines.append("**Contexto técnico**")
    if bug:
        lines.append(f"- Bug: {_text(bug)} [{', '.join(_source_ids(bug))}]")
        for detail in _details(bug):
            lines.append(f"- {detail.strip()}")
    if decision:
        lines.append(f"- Decisão: {_text(decision)} [{', '.join(_source_ids(decision))}]")
        if decision.get("reason"):
            lines.append(f"- Motivo: {decision['reason']}")
    for test in tests[-1:]:
        command = f" (comando: `{test['command']}`)" if test.get("command") else ""
        lines.append(f"- Teste observado: {test.get('text')}{command} [{', '.join(_source_ids(test))}]")
    if not (bug or decision or tests):
        lines.append(f"- Tarefa atual: {task.get('text') or 'não registrada'}")

    lines.append("")
    lines.append("**Restrições**")
    restrictions = list(decision.get("restrictions", ()) or ())
    if bug:
        restrictions.extend(bug.get("restrictions", ()) or ())
    if restrictions:
        lines.extend(f"- {item}" for item in restrictions)
    else:
        lines.append("- Respeitar o isolamento do projeto e as decisões vigentes.")

    lines.append("")
    lines.append("**Critérios de aceitação**")
    commands = [test["command"] for test in tests if test.get("command")]
    commands.extend(p["command"] for p in pending if p.get("command"))
    if bug and bug.get("command"):
        commands.append(bug["command"])
    if commands:
        lines.extend(f"- [ ] Rodar `{command}` e colar o resultado." for command in dict.fromkeys(commands))
    else:
        lines.append("- [ ] Rodar o teste do componente e colar o comando e o resultado.")
    if tests:
        lines.append("- [ ] Manter o comportamento coberto pelo teste observado.")
    if pending:
        lines.append(f"- [ ] Resolver: {pending[-1].get('text')}")
    if fields.get("gaps"):
        lines.append("- [ ] Confirmar: " + "; ".join(fields["gaps"]))
    return "\n".join(lines)
