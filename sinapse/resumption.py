"""Cenário de retomada de projeto: histórico recente vs política Sinapse.

Compara o histórico recente, a política atual e uma variante que trata
retomadas abertas como pedidos com múltiplos propósitos. A variante roda o
mesmo seletor congelado para decisão, teste concluído e pendência e recombina
as evidências sob o mesmo orçamento de memória, com as mesmas restrições de
sensibilidade.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter, render_evidence_block, wrap_memory
from .chat import EstimatedChatCounter, build_base_messages, with_memory
from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .models import Message, load_messages
from .scale_benchmark import _id_in_messages
from .sensitivity import SensitivityPolicy
from .text import infer_dimension, normalize_text
from .whiteboard import MemoryAgent

DATASET = "benchmark/resumption_sessions.json"
CONDITIONS = ("history", "sinapse", "sinapse_multipurpose")
ELEMENTS = ("decision", "reason", "completed", "pending")
PURPOSE_QUERIES = (
    ("decision", "Qual é a decisão vigente?"),
    ("completed", "Qual foi o último teste concluído?"),
    ("pending", "Qual é o próximo passo pendente?"),
)
PENDING_ACK = ("nao defini", "sem proximo passo", "nao ha proximo", "nenhum proximo")


@dataclass(frozen=True)
class ResumptionCase:
    id: str
    conversation_id: str
    question: str
    recent: tuple[str, ...]
    recent_ids: tuple[str, ...]
    expected_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    reason_ids: tuple[str, ...]
    completed_ids: tuple[str, ...]
    pending_ids: tuple[str, ...]
    decision_facts: tuple[str, ...]
    reason_facts: tuple[str, ...]
    completed_facts: tuple[str, ...]
    pending_facts: tuple[str, ...]
    components: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    completed_repetition_terms: tuple[str, ...]
    wrong_current_terms: tuple[str, ...]
    reference_answer: str
    rubric: str
    open_resumption: bool = False
    no_pending: bool = False
    memory_token_limit: int = 160
    preferences: tuple[str, ...] = ()
    condition_ids: tuple[str, ...] = ()
    condition_facts: tuple[str, ...] = ()
    old_task_ids: tuple[str, ...] = ()
    old_task_terms: tuple[str, ...] = ()
    wrong_project_ids: tuple[str, ...] = ()
    wrong_project_terms: tuple[str, ...] = ()


def cases() -> tuple[ResumptionCase, ...]:
    norte_recent = (
        "Passei a semana ajustando a iluminação.",
        "Depois volto ao problema do Pigments e da introdução.",
    )
    norte_ids = ("pr9", "pr10")
    serra_recent = ("Passei a semana fora do estúdio.", "Depois retomo o Serra e o Nova.")
    serra_ids = ("sr7", "sr8")
    litoral_recent = ("Voltei ao Litoral depois do intervalo.", "Depois continuo o Litoral e o Lyra.")
    litoral_ids = ("lt7", "lt8")
    vale_recent = ("Passei a semana organizando samples.", "Depois volto ao Vale e ao Comet.")
    vale_ids = ("vl5", "vl6")
    return (
        ResumptionCase(
            "retomada_geral",
            "c1",
            "Onde paramos nesse problema?",
            norte_recent,
            norte_ids,
            ("pr4", "pr5", "pr6", "pr7"),
            ("pr4",),
            ("pr4",),
            ("pr5", "pr8"),
            ("pr6", "pr7"),
            ("Pigments",),
            ("CPU", "40"),
            ("refrao",),
            ("introducao", "96 khz"),
            ELEMENTS,
            ("prazo", "cliente", "lançamento", "10%", "novo plugin"),
            (
                "repetir o teste do refrão",
                "testar o refrão de novo",
                "falta testar o refrão",
                "próximo teste é no refrão",
                "repetir a introdução com o Aurora",
            ),
            ("o preset é o Aurora", "voltei para o Aurora", "uso o Aurora"),
            (
                "A decisão vigente é o Pigments para o bass, porque a CPU do "
                "Aurora chegava a 40%. O último teste concluído foi o Pigments no "
                "refrão, com a CPU em 22%. A pendência é testar a introdução em "
                "96 kHz, para confirmar que a CPU não estoura."
            ),
            "Decisão atual, motivo, último teste concluído e pendência real; "
            "repetir passo concluído ou afirmar sem suporte desconta.",
            open_resumption=True,
        ),
        ResumptionCase(
            "proximo_teste",
            "c1",
            "Qual era o próximo teste e por quê?",
            norte_recent,
            norte_ids,
            ("pr6", "pr7"),
            (),
            ("pr4", "pr7"),
            ("pr5", "pr8"),
            ("pr6", "pr7"),
            (),
            ("CPU", "estoura"),
            ("refrao",),
            ("introducao", "96 khz"),
            ("reason", "pending"),
            ("prazo", "cliente", "lançamento", "10%", "novo plugin"),
            (
                "repetir o teste do refrão",
                "testar o refrão de novo",
                "falta testar o refrão",
                "próximo teste é no refrão",
                "repetir a introdução com o Aurora",
            ),
            ("o preset é o Aurora", "voltei para o Aurora", "uso o Aurora"),
            (
                "O próximo teste é a introdução em 96 kHz, para confirmar que a "
                "CPU não estoura."
            ),
            "Pendência correta e motivo; repetir o refrão, já concluído, "
            "desconta.",
        ),
        ResumptionCase(
            "decisao_vigente",
            "c1",
            "Qual preset está valendo para o bass agora?",
            norte_recent,
            norte_ids,
            ("pr4",),
            ("pr4",),
            ("pr4",),
            ("pr5", "pr8"),
            (),
            ("Pigments",),
            ("CPU", "40"),
            ("refrao",),
            (),
            ("decision", "reason"),
            ("prazo", "cliente", "lançamento", "10%", "novo plugin"),
            (
                "repetir o teste do refrão",
                "testar o refrão de novo",
                "falta testar o refrão",
            ),
            ("o preset é o Aurora", "voltei para o Aurora", "uso o Aurora"),
            (
                "A decisão vigente é o Pigments para o bass, porque a CPU do "
                "Aurora chegava a 40%."
            ),
            "Decisão vigente e motivo; citar o Aurora como atual desconta.",
        ),
        ResumptionCase(
            "pendencia_concluida",
            "serra",
            "Onde paramos no projeto Serra?",
            serra_recent,
            serra_ids,
            ("sr2", "sr5", "sr6"),
            ("sr2",),
            ("sr2",),
            ("sr5",),
            ("sr6",),
            ("Nova",),
            ("CPU", "alta"),
            ("ponte", "48 khz"),
            ("exportar",),
            ELEMENTS,
            (
                "falta testar a ponte",
                "a ponte ainda não",
                "prazo",
                "cliente",
            ),
            ("testar a ponte de novo", "repetir a ponte", "falta testar a ponte"),
            ("o preset é o Vega", "uso o Vega", "voltei para o Vega"),
            (
                "A decisão vigente é o Nova para o bass, porque a CPU do Vega "
                "ficava alta. O último teste concluído foi a ponte em 48 kHz. "
                "O próximo passo é exportar a versão final."
            ),
            "Decisão, motivo, teste concluído e pendência atual; a ponte já "
            "concluída não deve voltar como pendente.",
            open_resumption=True,
        ),
        ResumptionCase(
            "pendencia_substituida",
            "litoral",
            "Onde paramos no projeto Litoral?",
            litoral_recent,
            litoral_ids,
            ("lt2", "lt3", "lt5", "lt6"),
            ("lt2",),
            ("lt2",),
            ("lt3",),
            ("lt5", "lt6"),
            ("Lyra",),
            ("CPU", "alta"),
            ("introducao",),
            ("ponte", "48 khz"),
            ELEMENTS,
            (
                "falta testar o solo",
                "proximo passo e o solo",
                "o solo ainda",
                "prazo",
                "cliente",
            ),
            ("testar o solo de novo", "repetir o solo", "falta testar o solo"),
            ("o preset é o Orion", "uso o Orion", "voltei para o Orion"),
            (
                "A decisão vigente é o Lyra para a guitarra, porque a CPU do "
                "Orion ficava alta. O último teste concluído foi o Lyra na "
                "introdução. O próximo passo é testar a ponte em 48 kHz; o solo "
                "saiu do escopo."
            ),
            "Pendência substituída pela ponte; o solo antigo não deve voltar "
            "como próximo passo.",
            open_resumption=True,
        ),
        ResumptionCase(
            "sem_pendencia",
            "vale",
            "Onde paramos no projeto Vale?",
            vale_recent,
            vale_ids,
            ("vl2", "vl3", "vl4"),
            ("vl2",),
            ("vl2",),
            ("vl3",),
            ("vl4",),
            ("Comet",),
            ("CPU", "alta"),
            ("refrao",),
            PENDING_ACK,
            ELEMENTS,
            ("exportar", "testar a ponte", "novo teste", "prazo", "cliente"),
            ("repetir o teste do refrão", "testar o refrão de novo"),
            ("o preset é o Atlas", "uso o Atlas", "voltei para o Atlas"),
            (
                "A decisão vigente é o Comet, porque a CPU do Atlas ficava alta. "
                "O último teste concluído foi o Comet no refrão. Não há próximo "
                "passo definido para o Vale."
            ),
            "Decisão, motivo e teste concluído; a ausência de próximo passo "
            "deve ser reconhecida, sem inventar pendência.",
            open_resumption=True,
            no_pending=True,
        ),
        ResumptionCase(
            "pendencia_substituida_costa",
            "costa",
            "Onde paramos no projeto Costa?",
            ("Voltei ao Costa depois do intervalo.", "Depois continuo o Costa e o Vela."),
            ("cs7", "cs8"),
            ("cs2", "cs3", "cs5", "cs6"),
            ("cs2",),
            ("cs2",),
            ("cs3",),
            ("cs5", "cs6"),
            ("Vela",),
            ("CPU", "alta"),
            ("verso",),
            ("ponte", "48 khz"),
            ELEMENTS,
            (
                "testar o arpejo",
                "arpejo ainda",
                "proximo passo e o arpejo",
                "prazo",
            ),
            ("testar o arpejo de novo", "repetir o arpejo", "falta testar o arpejo"),
            ("o preset é o Íris", "uso o Íris", "voltei para o Íris"),
            (
                "A decisão vigente é o Vela, porque a CPU do Íris ficava alta. "
                "O último teste concluído foi o Vela no verso. O próximo passo é "
                "testar a ponte em 48 kHz; o arpejo saiu do escopo."
            ),
            "Pendência substituída pela ponte; o arpejo antigo não deve voltar "
            "como próximo passo.",
            open_resumption=True,
            old_task_ids=("cs4",),
            old_task_terms=(
                "testar o arpejo",
                "arpejo ainda",
                "proximo passo e o arpejo",
            ),
        ),
        ResumptionCase(
            "proximo_condicionado",
            "dunas",
            "Onde paramos no projeto Dunas?",
            ("Voltei ao Dunas depois do intervalo.", "Depois continuo o Dunas e o Aster."),
            ("dn7", "dn8"),
            ("dn2", "dn3", "dn4"),
            ("dn2",),
            ("dn2",),
            ("dn3",),
            ("dn4",),
            ("Aster",),
            ("CPU", "alta"),
            ("refrao",),
            ("ponte", "96 khz"),
            ELEMENTS,
            (
                "já exportei",
                "resultado aprovado",
                "cpu ficou abaixo",
                "prazo",
            ),
            ("repetir o refrão", "testar o refrão de novo"),
            ("o preset é o Halo", "uso o Halo", "voltei para o Halo"),
            (
                "A decisão vigente é o Aster, porque a CPU do Halo ficava alta. "
                "O último teste concluído foi o Aster no refrão. O próximo passo "
                "é testar a ponte em 96 kHz; se a CPU ficar abaixo de 30%, "
                "exportar; se não, reduzir o oversampling."
            ),
            "Pendência condicionada a um resultado desconhecido; a condição "
            "precisa aparecer junto do próximo passo.",
            open_resumption=True,
            condition_ids=("dn4",),
            condition_facts=("30", "exportar", "oversampling"),
        ),
        ResumptionCase(
            "intercalado_norte",
            "norte2",
            "Onde paramos no projeto Norte?",
            ("Voltei ao Norte depois do intervalo.", "Depois continuo o Norte e o Safira."),
            ("n5", "n6"),
            ("n2", "n3", "n4"),
            ("n2",),
            ("n2",),
            ("n3",),
            ("n4",),
            ("Safira",),
            ("CPU", "alta"),
            ("refrao",),
            ("ponte", "48 khz"),
            ELEMENTS,
            ("prazo", "cliente"),
            ("testar a ponte do Sul", "repetir o teste do Sul"),
            ("o preset é o Lume", "uso o Lume"),
            (
                "A decisão vigente é o Safira, porque a CPU do Lume ficava alta. "
                "O último teste concluído foi o Safira no refrão. Falta testar a "
                "ponte do Norte em 48 kHz."
            ),
            "Projeto Norte; tarefas do Sul não podem ser atribuídas ao Norte.",
            open_resumption=True,
            wrong_project_ids=("s2", "s3", "s4"),
            wrong_project_terms=("projeto sul", "no sul", "do sul"),
        ),
        ResumptionCase(
            "intercalado_sul",
            "sul",
            "Onde paramos no projeto Sul?",
            ("Voltei ao Sul depois do intervalo.", "Depois continuo o Sul e o Safira."),
            ("s5", "s6"),
            ("s2", "s3", "s4"),
            ("s2",),
            ("s2",),
            ("s3",),
            ("s4",),
            ("Safira",),
            ("CPU", "alta"),
            ("ponte", "48 khz"),
            ("refrao",),
            ELEMENTS,
            ("prazo", "cliente"),
            ("testar a ponte do Norte", "repetir o teste do Norte"),
            ("o preset é o Lume", "uso o Lume"),
            (
                "A decisão vigente é o Safira, porque a CPU do Lume ficava alta. "
                "O último teste concluído foi o Safira na ponte do Sul em 48 kHz. "
                "Falta testar o refrão do Sul."
            ),
            "Projeto Sul; tarefas do Norte não podem ser atribuídas ao Sul.",
            open_resumption=True,
            wrong_project_ids=("n2", "n3", "n4"),
            wrong_project_terms=("projeto norte", "no norte", "do norte"),
        ),
    )


def _context(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(item["content"]) for item in messages[:-1])


class SimulatedResumptionClient:
    """Responde com a evidência presente; valida casos, rastreamento e rubrica."""

    name = "simulated-resumption"
    simulated = True

    def __init__(
        self,
        case: ResumptionCase,
        messages: Sequence[Mapping[str, str]],
        corpus: Sequence[Message],
    ) -> None:
        self.case = case
        self.messages = messages
        self.by_id = {message.id: message for message in corpus}
        self.requests: list[dict[str, Any]] = []

    def _present(self, message_id: str) -> bool:
        message = self.by_id[message_id]
        if _id_in_messages(message_id, self.messages[:-1]):
            return True
        return normalize_text(message.text) in normalize_text(
            _context(self.messages)
        )

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse:
        self.requests.append(
            json.loads(json.dumps(request_body, ensure_ascii=False))
        )
        parts: list[str] = []
        decisions = [
            message_id
            for message_id in self.case.decision_ids
            if self._present(message_id)
        ]
        completed = [
            message_id
            for message_id in self.case.completed_ids
            if self._present(message_id)
        ]
        if decisions:
            parts.append(
                "Decisão vigente: " + self.by_id[decisions[-1]].text
            )
        if completed:
            parts.append(
                "Último teste concluído: " + self.by_id[completed[-1]].text
            )
        if self.case.no_pending:
            if any(self._present(message_id) for message_id in self.case.pending_ids):
                parts.append("Não há próximo passo definido para este projeto.")
        else:
            pending = [
                message_id
                for message_id in self.case.pending_ids
                if self._present(message_id)
            ]
            for message_id in pending:
                prefix = (
                    "Condição: "
                    if message_id in self.case.condition_ids
                    else "Próximo passo pendente: "
                )
                parts.append(prefix + self.by_id[message_id].text)
        answer = (
            " ".join(parts) if parts else "Não tenho informação suficiente."
        )
        return ChatCompletionResponse(
            content=answer,
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={"simulated": True},
        )


def evaluate_answer(
    case: ResumptionCase, answer: str
) -> tuple[float, dict[str, Any]]:
    normalized = normalize_text(answer)
    decision = any(
        normalize_text(fact) in normalized for fact in case.decision_facts
    ) and not any(
        normalize_text(term) in normalized for term in case.wrong_current_terms
    ) and any(
        marker in normalized
        for marker in (
            "decisao vigente",
            "agora uso",
            "uso o",
            "usa o",
            "estou usando",
            "troquei para",
            "preset atual",
        )
    )
    reason = any(
        normalize_text(fact) in normalized for fact in case.reason_facts
    ) and any(
        term in normalized
        for term in ("cpu", "40", "alta", "caiu", "22", "estoura")
    )
    completed = any(
        normalize_text(fact) in normalized for fact in case.completed_facts
    )
    pending = any(
        normalize_text(fact) in normalized for fact in case.pending_facts
    )
    checks = {
        "decision": decision,
        "reason": reason,
        "completed": completed,
        "pending": pending,
    }
    repetition = [
        term
        for term in case.completed_repetition_terms
        if normalize_text(term) in normalized
    ]
    unsupported = [
        claim
        for claim in case.forbidden_claims
        if normalize_text(claim) in normalized
    ]
    conditions_present = [
        fact
        for fact in case.condition_facts
        if normalize_text(fact) in normalized
    ]
    conditions_omitted = [
        fact
        for fact in case.condition_facts
        if normalize_text(fact) not in normalized
    ]
    old_task_terms = [
        term
        for term in case.old_task_terms
        if normalize_text(term) in normalized
    ]
    wrong_project_terms = [
        term
        for term in case.wrong_project_terms
        if normalize_text(term) in normalized
    ]
    components = [checks[name] for name in case.components]
    passed = sum(components) / len(components) if components else 1.0
    score = max(
        0.0,
        min(
            1.0,
            passed - 0.5 * len(repetition) - 0.5 * len(unsupported),
        ),
    )
    required = {
        "decision": case.decision_facts,
        "reason": case.reason_facts,
        "completed": case.completed_facts,
        "pending": case.pending_facts,
    }
    return round(score, 4), {
        **checks,
        "components": list(case.components),
        "facts_required": {
            name: list(facts)
            for name, facts in required.items()
            if name in case.components
        },
        "completed_repetition": repetition,
        "unsupported_claims": unsupported,
        "conditions_present": conditions_present,
        "conditions_omitted": conditions_omitted,
        "old_task_terms_present": old_task_terms,
        "wrong_project_terms_present": wrong_project_terms,
    }


def _present_ids(
    case: ResumptionCase,
    messages: Sequence[Mapping[str, str]],
    corpus: Sequence[Message],
) -> tuple[str, ...]:
    by_id = {message.id: message for message in corpus}
    context = _context(messages)
    present = []
    for message_id in case.expected_ids:
        message = by_id[message_id]
        if _id_in_messages(message_id, messages[:-1]):
            present.append(message_id)
        elif normalize_text(message.text) in normalize_text(context):
            present.append(message_id)
    return tuple(present)


def _references(
    case: ResumptionCase, selected: Sequence[str]
) -> dict[str, bool]:
    chosen = set(selected)
    names = list(ELEMENTS) + (
        ["condition"] if case.condition_ids else []
    )
    return {
        name: bool(chosen & set(getattr(case, f"{name}_ids")))
        for name in names
        if getattr(case, f"{name}_ids")
    }


SUPPORT_MARKERS = {
    "decision": ("preset", "troquei", "uso", "decidi", "escolhi"),
    "reason": ("porque", "cpu", "causa", "alta", "peso"),
    "completed": ("conclu", "aprovado", "terminei", "finalizado", "teste"),
    "pending": ("falta", "proximo", "pendente", "ainda", "depende"),
    "condition": ("se ", "caso", "depende", "abaixo", "senao", "se nao"),
}


def _reference_support(
    case: ResumptionCase,
    corpus: Sequence[Message],
    selected: Sequence[str],
) -> tuple[dict[str, bool], list[str]]:
    by_id = {message.id: message for message in corpus}
    chosen = set(selected)
    names = list(ELEMENTS) + (
        ["condition"] if case.condition_ids else []
    )
    support: dict[str, bool] = {}
    failures: list[str] = []
    for name in names:
        ids = getattr(case, f"{name}_ids")
        if not ids:
            continue
        picked = [message_id for message_id in ids if message_id in chosen]
        ok = bool(picked)
        for message_id in picked:
            text = normalize_text(by_id[message_id].text)
            if not any(marker in text for marker in SUPPORT_MARKERS[name]):
                ok = False
                failures.append(f"{name}:{message_id}")
        support[name] = ok
    return support, failures


def _window(
    case: ResumptionCase,
    instructions: str,
    settings: DeepSeekSettings,
) -> int:
    recent = tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(case.recent, case.recent_ids)
    )
    counter = EstimatedChatCounter(SimulatedCounter())
    base = build_base_messages(instructions, recent, case.question)
    return counter.count_messages(base) + settings.output_reserve + case.memory_token_limit


def _run_agent(
    case: ResumptionCase,
    corpus: Sequence[Message],
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
    question: str | None = None,
):
    recent = tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(case.recent, case.recent_ids)
    )
    agent = MemoryAgent(corpus, config)
    for preference in case.preferences:
        agent.declare_preference(case.conversation_id, preference, 0)
    return agent.before_response(
        case.conversation_id,
        question or case.question,
        recent,
        window=_window(case, str(config.get("instructions", "")), settings),
        output_reserve=settings.output_reserve,
        base_message_ids=case.recent_ids,
    )


def _memory_ids(turn) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            ((turn.seed.id,) if turn.budget.diagnostics.seed_included else ())
            + tuple(turn.budget.diagnostics.selected_ids)
        )
    )


def _multipurpose_context(
    case: ResumptionCase,
    corpus: Sequence[Message],
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...], int, list[dict[str, Any]]]:
    instructions = str(config.get("instructions", ""))
    recent = tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(case.recent, case.recent_ids)
    )
    base = build_base_messages(instructions, recent, case.question)
    counter = EstimatedChatCounter(SimulatedCounter())
    window = _window(case, instructions, settings)
    capacity = window - settings.output_reserve
    by_id = {message.id: message for message in corpus}
    policy = SensitivityPolicy()
    for preference in case.preferences:
        policy.declare(case.conversation_id, preference, 0)
    blocked = set(
        policy.evaluate(case.conversation_id, case.question, corpus, 0).blocked_ids
    )
    blocks: list[str] = []
    chosen: list[str] = []
    passes: list[dict[str, Any]] = []
    for name, purpose in PURPOSE_QUERIES:
        turn = _run_agent(
            case, corpus, config, settings, f"{case.question} {purpose}"
        )
        ids = [
            message_id
            for message_id in dict.fromkeys(
                (turn.seed.id,) + tuple(turn.budget.diagnostics.selected_ids)
            )
            if message_id in by_id and message_id not in blocked
        ]
        passes.append(
            {
                "purpose": name,
                "query": f"{case.question} {purpose}",
                "seed": turn.seed.id,
                "ids": ids,
            }
        )
        for message_id in ids:
            if message_id in chosen:
                continue
            block = render_evidence_block(by_id[message_id], len(chosen) + 1)
            trial = with_memory(
                base, wrap_memory(blocks + [block])
            )
            if counter.count_messages(trial) <= capacity:
                blocks.append(block)
                chosen.append(message_id)
    messages = with_memory(base, wrap_memory(blocks))
    memory_tokens = counter.count_messages(messages) - counter.count_messages(base)
    return messages, tuple(chosen), memory_tokens, passes


def _run_condition(
    case: ResumptionCase,
    condition: str,
    corpus: Sequence[Message],
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
    client: ChatTransport | None,
) -> dict[str, Any]:
    instructions = str(config.get("instructions", ""))
    recent = tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(case.recent, case.recent_ids)
    )
    counter = EstimatedChatCounter(SimulatedCounter())
    window = _window(case, instructions, settings)
    memory_corpus = [
        message for message in corpus if message.id not in case.recent_ids
    ]
    memory_tokens = 0
    selected: tuple[str, ...] = ()
    passes: list[dict[str, Any]] = []
    started = time.perf_counter()
    if condition == "history":
        messages = build_base_messages(instructions, recent, case.question)
    elif condition == "sinapse_multipurpose" and case.open_resumption:
        messages, selected, memory_tokens, passes = _multipurpose_context(
            case, memory_corpus, config, settings
        )
    else:
        turn = _run_agent(case, memory_corpus, config, settings)
        messages = turn.messages
        memory_tokens = turn.budget.diagnostics.memory_tokens
        selected = _memory_ids(turn)
    elapsed = (time.perf_counter() - started) * 1000
    request = build_request_body(messages, settings)
    answer_client = (
        client
        if client is not None
        else SimulatedResumptionClient(case, messages, corpus)
    )
    error = None
    answer = ""
    usage: Mapping[str, Any] = {}
    try:
        response: ChatCompletionResponse = answer_client.complete(
            request, timeout=settings.timeout
        )
        answer = response.content
        usage = response.usage
    except Exception as exc:  # pragma: no cover - falha de transporte real
        error = str(exc)
    score, checks = evaluate_answer(case, answer)
    present = _present_ids(case, messages, corpus)
    references = _references(case, selected)
    support, support_failures = _reference_support(case, corpus, selected)
    relevant = list(case.components) + (
        ["condition"] if case.condition_ids else []
    )
    support_all = all(support.get(name, False) for name in relevant)
    old_selected = [
        message_id for message_id in case.old_task_ids if message_id in selected
    ]
    wrong_selected = [
        message_id
        for message_id in case.wrong_project_ids
        if message_id in selected
    ]
    old_tasks = sorted(
        set(old_selected) | set(checks["old_task_terms_present"])
    )
    wrong_project = sorted(
        set(wrong_selected) | set(checks["wrong_project_terms_present"])
    )
    errors: list[str] = [
        f"faltou:{name}"
        for name in case.components
        if not checks[name]
    ]
    if checks["conditions_omitted"]:
        errors.append(
            "condicao_omitida:" + ",".join(checks["conditions_omitted"])
        )
    if old_tasks:
        errors.append("tarefa_antiga_como_atual:" + ",".join(old_tasks))
    if wrong_project:
        errors.append("projeto_errado:" + ",".join(wrong_project))
    if checks["completed_repetition"]:
        errors.append(
            "repetiu_concluido:" + ",".join(checks["completed_repetition"])
        )
    if checks["unsupported_claims"]:
        errors.append(
            "sem_suporte:" + ",".join(checks["unsupported_claims"])
        )
    if support_failures:
        errors.append("referencia_sem_suporte:" + ",".join(support_failures))
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = counter.count_messages(messages)
    if completion_tokens is None:
        completion_tokens = SimulatedCounter().count(answer)
    return {
        "case": case.id,
        "condition": condition,
        "question": case.question,
        "open_resumption": case.open_resumption,
        "has_conditions": bool(case.condition_facts),
        "selected_ids": list(selected),
        "multipurpose_passes": passes,
        "evidence_present": list(present),
        "evidence_coverage": (
            round(len(present) / len(case.expected_ids), 4)
            if case.expected_ids
            else 1.0
        ),
        "element_checks": {
            name: checks[name] for name in case.components
        },
        "references": references,
        "references_all": all(references.values()) if references else True,
        "reference_support": support,
        "reference_support_failures": support_failures,
        "reference_support_all": support_all,
        "old_tasks_as_current": old_tasks,
        "wrong_project": wrong_project,
        "errors": errors,
        "answer": answer,
        "score": score,
        "checks": checks,
        "memory_tokens": memory_tokens,
        "select_ms": round(elapsed, 2),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "token_source": "api" if usage else "estimativa_local",
        "error": error,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    element_rows = {
        name: [row for row in rows if name in row["checks"]["components"]]
        for name in ELEMENTS
    }
    condition_rows = [row for row in rows if row["has_conditions"]]
    return {
        "queries": count,
        **{
            f"{name}_ok": sum(
                row["element_checks"].get(name, False)
                for row in element_rows[name]
            )
            for name in ELEMENTS
        },
        **{
            f"{name}_total": len(element_rows[name])
            for name in ELEMENTS
        },
        **{
            f"ref_{name}": sum(
                row["references"].get(name, False) for row in rows
            )
            for name in (*ELEMENTS, "condition")
        },
        "references_all": sum(row["references_all"] for row in rows),
        "reference_support_all": sum(
            row["reference_support_all"] for row in rows
        ),
        "conditions_ok": sum(
            not row["checks"]["conditions_omitted"]
            for row in condition_rows
        ),
        "conditions_total": len(condition_rows),
        "conditions_omitted": sum(
            bool(row["checks"]["conditions_omitted"])
            for row in condition_rows
        ),
        "old_tasks_as_current": sum(
            bool(row["old_tasks_as_current"]) for row in rows
        ),
        "wrong_project": sum(bool(row["wrong_project"]) for row in rows),
        "errors": sum(len(row["errors"]) for row in rows),
        "coverage": round(
            sum(
                sum(row["element_checks"].values())
                / len(row["element_checks"])
                for row in rows
            )
            / count,
            4,
        ),
        "evidence_coverage": round(
            sum(row["evidence_coverage"] for row in rows) / count, 4
        ),
        "avg_score": round(sum(row["score"] for row in rows) / count, 4),
        "completed_repetitions": sum(
            bool(row["checks"]["completed_repetition"]) for row in rows
        ),
        "unsupported_claims": sum(
            len(row["checks"]["unsupported_claims"]) for row in rows
        ),
        "prompt_tokens_total": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens_total": sum(
            row["completion_tokens"] for row in rows
        ),
        "memory_tokens_avg": round(
            sum(row["memory_tokens"] for row in rows) / count, 1
        ),
        "select_ms_median": round(
            sorted(row["select_ms"] for row in rows)[count // 2], 2
        ),
    }


def _trace(
    case: ResumptionCase,
    corpus: Sequence[Message],
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
    multipurpose: bool,
) -> dict[str, Any]:
    memory_corpus = [
        message for message in corpus if message.id not in case.recent_ids
    ]
    focus, rule = infer_dimension(case.question)
    if multipurpose and case.open_resumption:
        _, selected, memory_tokens, passes = _multipurpose_context(
            case, memory_corpus, config, settings
        )
        return {
            "case": case.id,
            "condition": "sinapse_multipurpose",
            "question": case.question,
            "focus": focus,
            "focus_rule": rule,
            "open_resumption": case.open_resumption,
            "purpose_inferred": "multiplos_propositos",
            "passes": passes,
            "selected": list(selected),
            "memory_tokens": memory_tokens,
        }
    turn = _run_agent(case, memory_corpus, config, settings)
    decisions = {item.id: item for item in turn.budget.diagnostics.decisions}
    element_ids = set().union(
        *(
            set(getattr(case, f"{name}_ids"))
            for name in ELEMENTS
        )
    )
    candidates = []
    for candidate in turn.recall.candidates:
        if candidate.message.id not in element_ids:
            continue
        decision = decisions.get(candidate.message.id)
        candidates.append(
            {
                "id": candidate.message.id,
                "within_radius": candidate.within_radius,
                "attention": round(candidate.attention, 4),
                "distance": round(candidate.distance, 4),
                "evidence_focus": candidate.evidence.get(turn.recall.focus),
                "reason": decision.reason if decision else None,
                "detail": decision.detail if decision else None,
            }
        )
    return {
        "case": case.id,
        "condition": "sinapse_multipurpose" if multipurpose else "sinapse",
        "question": case.question,
        "focus": focus,
        "focus_rule": rule,
        "open_resumption": case.open_resumption,
        "purpose_inferred": "foco_unico",
        "seed": turn.seed.id,
        "candidates": candidates,
        "selected": list(_memory_ids(turn)),
        "final_ids": [
            message.id
            for message in memory_corpus
            if _id_in_messages(message.id, turn.messages)
        ],
        "memory_tokens": turn.budget.diagnostics.memory_tokens,
        "decisions": {
            item.id: {
                "reason": item.reason,
                "detail": item.detail,
            }
            for item in turn.budget.diagnostics.decisions
            if item.id in element_ids
        },
    }


def run_resumption_benchmark(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
    path: str | Path = DATASET,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    corpus = load_messages(path)
    rows = []
    for case in cases():
        for condition in CONDITIONS:
            rows.append(
                _run_condition(
                    case, condition, corpus, config, settings, client
                )
            )
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in CONDITIONS
    }
    comparisons = []
    for case in cases():
        history = next(
            row for row in by_condition["history"] if row["case"] == case.id
        )
        sinapse = next(
            row for row in by_condition["sinapse"] if row["case"] == case.id
        )
        multipurpose = next(
            row
            for row in by_condition["sinapse_multipurpose"]
            if row["case"] == case.id
        )
        comparisons.append(
            {
                "case": case.id,
                "score_history": history["score"],
                "score_sinapse": sinapse["score"],
                "score_multipurpose": multipurpose["score"],
                "elements_sinapse": dict(sinapse["element_checks"]),
                "elements_multipurpose": dict(
                    multipurpose["element_checks"]
                ),
                "references_sinapse": dict(sinapse["references"]),
                "references_multipurpose": dict(
                    multipurpose["references"]
                ),
                "sinapse_better_history": sinapse["score"] > history["score"],
                "sinapse_worse_history": sinapse["score"] < history["score"],
                "multipurpose_better": multipurpose["score"] > sinapse["score"],
                "multipurpose_worse": multipurpose["score"] < sinapse["score"],
            }
        )
    traces = []
    for case_id in ("retomada_geral", "proximo_teste"):
        case = next(item for item in cases() if item.id == case_id)
        traces.append(_trace(case, corpus, config, settings, False))
        traces.append(_trace(case, corpus, config, settings, True))
    regression_ids = {case.id for case in cases()[:6]}
    new_ids = {case.id for case in cases()[6:]}
    regression_rows = [row for row in rows if row["case"] in regression_ids]
    new_rows = [row for row in rows if row["case"] in new_ids]
    return {
        "dataset": str(path),
        "mode": "real" if client is not None else "simulated",
        "model": settings.model,
        "cases": len(cases()),
        "rows": rows,
        "traces": traces,
        "comparisons": comparisons,
        "aggregates": {
            condition: _aggregate(group)
            for condition, group in by_condition.items()
        },
        "aggregates_regression": {
            condition: _aggregate(
                [row for row in regression_rows if row["condition"] == condition]
            )
            for condition in CONDITIONS
        },
        "aggregates_new": {
            condition: _aggregate(
                [row for row in new_rows if row["condition"] == condition]
            )
            for condition in CONDITIONS
        },
        "sinapse_better_history": sum(
            item["sinapse_better_history"] for item in comparisons
        ),
        "sinapse_worse_history": sum(
            item["sinapse_worse_history"] for item in comparisons
        ),
        "multipurpose_better": sum(
            item["multipurpose_better"] for item in comparisons
        ),
        "multipurpose_worse": sum(
            item["multipurpose_worse"] for item in comparisons
        ),
        "real_evaluation": {
            "available": client is not None,
            "model": settings.model if client is not None else None,
            "note": (
                "Avaliação real executada com o transporte DeepSeek."
                if client is not None
                else "Não executada nesta rodada: apenas cliente simulado."
            ),
        },
        "real_pairing": {
            "available": False,
            "model": settings.model,
            "conditions": ["sinapse", "sinapse_multipurpose"],
            "rubric": (
                "mesma rubrica: decisão vigente, motivo, último teste "
                "concluído, próximo passo, condição, referências, tarefa "
                "antiga como atual, atribuição ao projeto errado."
            ),
            "note": (
                "Comparação pareada preparada; respostas reais ainda não "
                "executadas."
            ),
        },
        "human_review_pending": True,
        "previous_review_preserved": "benchmark_out_resumption/resumption_review.json",
    }


def export_resumption(
    result: Mapping[str, Any], out_dir: str | Path
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resumption_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, row in enumerate(result["rows"], start=1):
        review.append(
            {
                "review_id": f"T{index:02d}",
                "case": row["case"],
                "condition": row["condition"],
                "question": row["question"],
                "answer": row["answer"],
                "rubric": next(
                    case.rubric for case in cases() if case.id == row["case"]
                ),
                "checks": row["checks"],
                "references": row["references"],
                "human_review": {
                    "reviewed": False,
                    "approved": None,
                    "notes": "",
                },
            }
        )
    (out / "resumption_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "RETOMADA DE PROJETO",
        f"modo={result['mode']} | casos={result['cases']}",
        "",
    ]
    for condition in CONDITIONS:
        item = result["aggregates"][condition]
        lines.append(
            f"{condition:<20} decisão={item['decision_ok']}/{item['decision_total']} "
            f"motivo={item['reason_ok']}/{item['reason_total']} "
            f"concluído={item['completed_ok']}/{item['completed_total']} "
            f"pendência={item['pending_ok']}/{item['pending_total']} "
            f"nota={item['avg_score']:.2f} "
            f"refs={item['references_all']} "
            f"repetições={item['completed_repetitions']} "
            f"sem_suporte={item['unsupported_claims']} "
            f"prompt={item['prompt_tokens_total']} "
            f"completion={item['completion_tokens_total']}"
        )
    (out / "resumption_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    pairs = []
    for case in cases():
        selected = {
            condition: next(
                row
                for row in result["rows"]
                if row["case"] == case.id and row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        pairs.append(
            {
                "case": case.id,
                "question": case.question,
                "rubric": case.rubric,
                "expected_ids": list(case.expected_ids),
                "condition_ids": list(case.condition_ids),
                "condition_facts": list(case.condition_facts),
                "old_task_ids": list(case.old_task_ids),
                "wrong_project_ids": list(case.wrong_project_ids),
                "history_selected": selected["history"]["selected_ids"],
                "sinapse_selected": selected["sinapse"]["selected_ids"],
                "multipurpose_selected": selected["sinapse_multipurpose"][
                    "selected_ids"
                ],
                "sinapse_answer": selected["sinapse"]["answer"],
                "multipurpose_answer": selected["sinapse_multipurpose"][
                    "answer"
                ],
                "real_answer": None,
                "status": "pending_real_deepseek",
            }
        )
    (out / "real_comparison.json").write_text(
        json.dumps(
            {
                "available": result["real_pairing"]["available"],
                "model": result["real_pairing"]["model"],
                "rubric": result["real_pairing"]["rubric"],
                "note": result["real_pairing"]["note"],
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "results": str(out / "resumption_results.json"),
        "review": str(out / "resumption_review.json"),
        "summary": str(out / "resumption_summary.txt"),
        "real_comparison": str(out / "real_comparison.json"),
    }
