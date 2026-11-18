"""Benchmark reproduzível da Sinapse.

Compara quatro condições com o mesmo modelo, instruções, histórico, pergunta,
reserva de saída e limite de memória:

- ``history``: histórico recente, sem memória adicional;
- ``lexical``: busca lexical por similaridade;
- ``sinapse``: evocação + quadro + atenção temática + utilidade marginal;
- ``manual``: evidências corretas escolhidas à mão (oráculo de seleção).

A avaliação de seleção usa o contexto efetivamente enviado. A avaliação de
resposta usa verificações automáticas e exporta rubricas para revisão humana
sem identificar a condição. O modo padrão usa um cliente simulado e serve
apenas para validar o mecanismo; ``--deepseek`` faz chamadas reais.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages, select_chat_context
from .chatbot import SinapseChatbot  # noqa: F401  (interface pública reutilizada)
from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekClient,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .memory import message_tokens, recall
from .models import DIMENSIONS, Candidate, Message, load_messages
from .sensitivity import DerivedSummary, SensitivityPolicy
from .text import cosine_overlap, normalize_text, tokenize
from .whiteboard import MemoryAgent

CONDITIONS = ("history", "lexical", "sinapse", "manual")
MISSING_MARKERS = (
    "nao tenho",
    "nao sei",
    "sem informacao",
    "nao consta",
    "nao disponivel",
    "nao ha informacao",
)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    kind: str
    messages: tuple[Message, ...]
    conversation_id: str
    recent: tuple[RecentTurn, ...]
    question: str
    expected_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    noise_ids: tuple[str, ...]
    distractor_ids: tuple[str, ...]
    memory_token_limit: int
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    must_acknowledge_missing: bool
    reference_answer: str
    rubric: str
    preferences: tuple[str, ...]
    summaries: tuple[DerivedSummary, ...]
    from_history: bool
    no_evidence: bool


def _recent_turn(raw: Any) -> RecentTurn:
    if isinstance(raw, str):
        return RecentTurn(text=raw)
    return RecentTurn(
        text=str(raw["text"]),
        id=raw.get("id"),
        author=str(raw.get("author", "user")),
        position=raw.get("position"),
    )


def load_cases(path: str | Path, root: str | Path | None = None) -> list[BenchmarkCase]:
    root = Path(root or ".")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for raw in data:
        messages: list[Message] = []
        for relative in raw.get("messages", []):
            messages.extend(load_messages(root / relative))
        ids = [message.id for message in messages]
        duplicated = sorted({message_id for message_id in ids if ids.count(message_id) > 1})
        if duplicated:
            raise ValueError(f"{raw['id']}: IDs duplicados em {duplicated}")
        cases.append(
            BenchmarkCase(
                id=str(raw["id"]),
                kind=str(raw.get("kind", "geral")),
                messages=tuple(messages),
                conversation_id=str(raw.get("conversation_id", "c1")),
                recent=tuple(_recent_turn(item) for item in raw.get("recent", [])),
                question=str(raw["question"]),
                expected_ids=tuple(raw.get("expected_ids", [])),
                dependency_ids=tuple(raw.get("dependency_ids", [])),
                forbidden_ids=tuple(raw.get("forbidden_ids", [])),
                forbidden_terms=tuple(raw.get("forbidden_terms", [])),
                noise_ids=tuple(raw.get("noise_ids", [])),
                distractor_ids=tuple(raw.get("distractor_ids", [])),
                memory_token_limit=int(raw.get("memory_token_limit", 140)),
                required_facts=tuple(raw.get("required_facts", [])),
                forbidden_claims=tuple(raw.get("forbidden_claims", [])),
                must_acknowledge_missing=bool(raw.get("must_acknowledge_missing", False)),
                reference_answer=str(raw.get("reference_answer", "Não tenho informação.")),
                rubric=str(raw.get("rubric", "")),
                preferences=tuple(raw.get("preferences", [])),
                summaries=tuple(
                    DerivedSummary(
                        id=str(summary.get("id", f"s{index}")),
                        conversation_id=str(raw.get("conversation_id", "c1")),
                        text=str(summary["text"]),
                        source_ids=tuple(summary.get("source_ids", [])),
                    )
                    for index, summary in enumerate(raw.get("summaries", []), start=1)
                ),
                from_history=bool(raw.get("from_history", False)),
                no_evidence=bool(raw.get("no_evidence", False)),
            )
        )
    return cases


@dataclass
class ConditionOutcome:
    condition: str
    messages: tuple[dict[str, str], ...]
    selected_ids: tuple[str, ...]
    evidence_present: tuple[str, ...]
    hit_any: bool
    full_coverage: bool
    additional_tokens: int
    used_tokens: int
    base_tokens: int
    restricted_violations: tuple[str, ...]
    repetition: bool
    irrelevant_attached: bool
    noise_selected: tuple[str, ...]
    answer: str
    answer_score: float
    checks: Mapping[str, Any]
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "selected_ids": list(self.selected_ids),
            "evidence_present": list(self.evidence_present),
            "hit_any": self.hit_any,
            "full_coverage": self.full_coverage,
            "additional_tokens": self.additional_tokens,
            "used_tokens": self.used_tokens,
            "base_tokens": self.base_tokens,
            "restricted_violations": list(self.restricted_violations),
            "repetition": self.repetition,
            "irrelevant_attached": self.irrelevant_attached,
            "noise_selected": list(self.noise_selected),
            "answer": self.answer,
            "answer_score": self.answer_score,
            "checks": dict(self.checks),
            "error": self.error,
        }


@dataclass
class CaseResult:
    case: BenchmarkCase
    outcomes: dict[str, ConditionOutcome]

    def as_dict(self) -> dict:
        return {
            "case": self.case.id,
            "kind": self.case.kind,
            "outcomes": {name: outcome.as_dict() for name, outcome in self.outcomes.items()},
        }


class SimulatedAnswerClient:
    """Cliente simulado: responde conforme a evidência presente na requisição.

    Serve para validar o mecanismo de avaliação. Não é o DeepSeek e seus
    resultados não devem ser atribuídos ao modelo real.
    """

    name = "simulated-benchmark"
    simulated = True

    def __init__(self, case: BenchmarkCase, messages: Sequence[Mapping[str, str]]) -> None:
        self.case = case
        self.messages = messages
        self.requests: list[dict] = []

    def _present(self, message_id: str) -> bool:
        context = "\n".join(str(item["content"]) for item in self.messages)
        message = next(
            (item for item in self.case.messages if item.id == message_id), None
        )
        if _id_marker(context, message_id):
            return True
        if message is not None and normalize_text(message.text) in normalize_text(context):
            return True
        return False

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse:
        self.requests.append(json.loads(json.dumps(request_body, ensure_ascii=False)))
        present = [message_id for message_id in self.case.expected_ids if self._present(message_id)]
        if not self.case.expected_ids:
            answer = self.case.reference_answer
        elif len(present) == len(self.case.expected_ids):
            answer = self.case.reference_answer
        elif present:
            answer = "Tenho apenas parte da informação."
        else:
            answer = "Não tenho informação suficiente."
        return ChatCompletionResponse(
            content=answer,
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={"simulated": True},
        )


def _normalized_contains(text: str, needle: str) -> bool:
    return normalize_text(needle) in normalize_text(text)


def _context_text(messages: Sequence[Mapping[str, str]]) -> str:
    # Preserva o turno atual; a checagem de conteúdo restrito ignora a última
    # mensagem (a pergunta do usuário).
    return "\n".join(str(message["content"]) for message in messages[:-1])


def _id_marker(context: str, message_id: str) -> bool:
    pattern = rf"(?<![a-z0-9_])id={re.escape(message_id)}(?!\d)"
    return bool(re.search(pattern, context)) or f"[{message_id}]" in context


def _present_ids(case: BenchmarkCase, messages: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    by_id = {message.id: message for message in case.messages}
    context = _context_text(messages)
    present: list[str] = []
    for message_id in case.expected_ids:
        message = by_id.get(message_id)
        if _id_marker(context, message_id):
            present.append(message_id)
            continue
        if message is not None and _normalized_contains(context, message.text):
            present.append(message_id)
    return tuple(present)


def _restricted_violations(
    case: BenchmarkCase, messages: Sequence[Mapping[str, str]]
) -> tuple[str, ...]:
    by_id = {message.id: message for message in case.messages}
    context = _context_text(messages)
    violations: list[str] = []
    for message_id in case.forbidden_ids:
        message = by_id.get(message_id)
        if _id_marker(context, message_id):
            violations.append(message_id)
        elif message is not None and _normalized_contains(context, message.text):
            violations.append(message_id)
    for term in case.forbidden_terms:
        if _normalized_contains(context, term):
            violations.append(f"termo:{normalize_text(term)}")
    return tuple(sorted(set(violations)))


def score_answer(case: BenchmarkCase, answer: str) -> tuple[float, dict[str, Any]]:
    facts = [fact for fact in case.required_facts if _normalized_contains(answer, fact)]
    fact_ratio = len(facts) / len(case.required_facts) if case.required_facts else 1.0
    forbidden = [claim for claim in case.forbidden_claims if _normalized_contains(answer, claim)]
    sensitive = [term for term in case.forbidden_terms if _normalized_contains(answer, term)]
    acknowledged = any(
        _normalized_contains(answer, marker) for marker in MISSING_MARKERS
    )
    missing_ok = (not case.must_acknowledge_missing) or acknowledged
    penalty = 0.5 * len(forbidden) + 0.5 * len(sensitive)
    if not missing_ok:
        penalty += 0.5
    score = max(0.0, min(1.0, fact_ratio - penalty))
    return round(score, 4), {
        "facts_present": facts,
        "facts_required": list(case.required_facts),
        "forbidden_claims_present": forbidden,
        "sensitive_terms_present": sensitive,
        "acknowledged_missing": acknowledged,
        "missing_required": case.must_acknowledge_missing,
    }


def _prepare(
    case: BenchmarkCase, config: Mapping[str, Any], settings: DeepSeekSettings
) -> tuple[SensitivityPolicy, tuple[RecentTurn, ...], str, int, int, int]:
    policy = SensitivityPolicy()
    for preference in case.preferences:
        policy.declare(case.conversation_id, preference, 0)
    allowed_summaries = [
        summary
        for summary in case.summaries
        if not policy.summary_is_blocked(case.conversation_id, summary, case.question)[0]
    ]
    summary_text = "\n\n".join(summary.text for summary in allowed_summaries)
    instructions = str(config.get("instructions", ""))
    if summary_text:
        instructions += "\n\n=== RESUMO (Sinapse) ===\n" + summary_text

    blocked = set(
        policy.evaluate(case.conversation_id, case.question, case.messages, 0).blocked_ids
    )
    recent: list[RecentTurn] = []
    for turn in case.recent:
        if turn.id and turn.id in blocked:
            continue
        if policy.text_is_blocked(case.conversation_id, turn.text, case.question)[0]:
            continue
        recent.append(turn)
    base = build_base_messages(instructions, tuple(recent), case.question)
    counter = EstimatedChatCounter(SimulatedCounter())
    base_tokens = counter.count_messages(base)
    window = base_tokens + settings.output_reserve + case.memory_token_limit
    return policy, tuple(recent), instructions, base_tokens, window, settings.output_reserve


def _lexical_candidates(
    case: BenchmarkCase, seed: Message, focus: str, blocked: set[str]
) -> list[Candidate]:
    query_tokens = tokenize(case.question)
    candidates: list[Candidate] = []
    for message in case.messages:
        if message.conversation_id != case.conversation_id:
            continue
        if message.id in blocked or message.id == seed.id:
            continue
        score = cosine_overlap(message_tokens(message), query_tokens)
        if score <= 0:
            continue
        evidence: dict[str, float | None] = {dimension: None for dimension in DIMENSIONS}
        evidence[focus] = 1.0
        candidates.append(
            Candidate(
                message=message,
                distance=1.0 - score,
                association=score,
                association_content=score,
                association_link=0.0,
                association_source="lexical",
                evidence=evidence,
                attention=score,
                within_radius=True,
            )
        )
    return candidates


def _manual_candidates(
    case: BenchmarkCase, message_ids: Sequence[str], focus: str, blocked: set[str]
) -> list[Candidate]:
    by_id = {message.id: message for message in case.messages}
    candidates: list[Candidate] = []
    for message_id in message_ids:
        if message_id in blocked or message_id not in by_id:
            continue
        evidence: dict[str, float | None] = {dimension: None for dimension in DIMENSIONS}
        evidence[focus] = 1.0
        candidates.append(
            Candidate(
                message=by_id[message_id],
                distance=0.0,
                association=1.0,
                association_content=1.0,
                association_link=0.0,
                association_source="manual",
                evidence=evidence,
                attention=1.0,
                within_radius=True,
            )
        )
    return candidates


def _memory_ids(seed: Message, diagnostics) -> tuple[str, ...]:
    ids: list[str] = []
    if diagnostics.seed_included:
        ids.append(seed.id)
    ids.extend(diagnostics.selected_ids)
    return tuple(dict.fromkeys(ids))


def _run_case(
    case: BenchmarkCase,
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
    real_client: ChatTransport | None,
) -> CaseResult:
    policy, recent, instructions, base_tokens, window, reserve = _prepare(
        case, config, settings
    )
    counter = EstimatedChatCounter(SimulatedCounter())
    blocked = set(
        policy.evaluate(case.conversation_id, case.question, case.messages, 0).blocked_ids
    )
    base = build_base_messages(instructions, recent, case.question)
    by_id = {message.id: message for message in case.messages}
    outcomes: dict[str, ConditionOutcome] = {}

    def evaluate_context(
        condition: str,
        messages: tuple[dict[str, str], ...],
        selected_ids: Sequence[str],
        memory_tokens: int,
    ) -> ConditionOutcome:
        present = _present_ids(case, messages)
        violations = _restricted_violations(case, messages)
        answer = ""
        error = None
        if real_client is not None:
            client: ChatTransport = real_client
        else:
            client = SimulatedAnswerClient(case, messages)
        try:
            response = client.complete(
                build_request_body(messages, settings), timeout=settings.timeout
            )
            answer = response.content
        except Exception as exc:  # erro de rede/API não interrompe os outros casos
            error = str(exc)
        score, checks = score_answer(case, answer)
        selected = tuple(selected_ids)
        return ConditionOutcome(
            condition=condition,
            messages=messages,
            selected_ids=selected,
            evidence_present=present,
            hit_any=bool(set(present) & set(case.expected_ids)),
            full_coverage=set(case.expected_ids) <= set(present),
            additional_tokens=int(memory_tokens),
            used_tokens=counter.count_messages(messages),
            base_tokens=base_tokens,
            restricted_violations=violations,
            repetition=bool(set(selected) & set(case.expected_ids)) and case.from_history,
            irrelevant_attached=bool(case.no_evidence and selected),
            noise_selected=tuple(sorted(set(selected) & set(case.noise_ids))),
            answer=answer,
            answer_score=score,
            checks=checks,
            error=error,
        )

    outcomes["history"] = evaluate_context("history", base, (), 0)

    # Busca lexical: mesma semente, mesmo orçamento, sem quadro/tema/utilidade.
    seed_result = recall(
        case.messages,
        case.question,
        [turn.text for turn in recent],
        config,
        conversation_id=case.conversation_id,
        excluded_ids=tuple(blocked),
    )
    lexical = select_chat_context(
        instructions=instructions,
        recent_turns=recent,
        current_turn=case.question,
        seed=seed_result.seed,
        focus=seed_result.focus,
        candidates=_lexical_candidates(case, seed_result.seed, seed_result.focus, blocked),
        window=window,
        output_reserve=reserve,
        counter=counter,
        top_k=int(config.get("top_k", 3)),
        threshold=float(config.get("attention_threshold", 0.05)),
        include_seed=True,
        use_utility=False,
    )
    outcomes["lexical"] = evaluate_context(
        "lexical",
        lexical.messages,
        _memory_ids(seed_result.seed, lexical.diagnostics),
        lexical.diagnostics.memory_tokens,
    )

    # Sinapse: agente com quadro, temas, sensibilidade e utilidade marginal.
    agent = MemoryAgent(case.messages, config)
    for preference in case.preferences:
        agent.declare_preference(case.conversation_id, preference, 0)
    for summary in case.summaries:
        agent.add_summary(
            summary.conversation_id, summary.text, summary.source_ids, summary.id
        )
    agent_turn = agent.before_response(
        case.conversation_id,
        case.question,
        recent,
        window=window,
        output_reserve=reserve,
    )
    outcomes["sinapse"] = evaluate_context(
        "sinapse",
        agent_turn.messages,
        _memory_ids(agent_turn.seed, agent_turn.budget.diagnostics),
        agent_turn.budget.diagnostics.memory_tokens,
    )

    # Manual: evidências corretas + dependências, sob o mesmo orçamento.
    if not case.expected_ids:
        outcomes["manual"] = evaluate_context("manual", base, (), 0)
        return CaseResult(case=case, outcomes=outcomes)
    manual_ids = tuple(
        dict.fromkeys(
            tuple(case.expected_ids)
            + tuple(
                dependency
                for dependency in case.dependency_ids
                if dependency not in case.expected_ids
            )
        )
    )
    seed_id = case.dependency_ids[0] if case.dependency_ids else None
    manual_recall = recall(
        case.messages,
        case.question,
        [turn.text for turn in recent],
        config,
        conversation_id=case.conversation_id,
        seed_id=seed_id,
        excluded_ids=tuple(blocked),
    )
    manual_seed = (
        by_id[seed_id]
        if seed_id and seed_id in by_id and seed_id not in blocked
        else manual_recall.seed
    )
    manual_focus = manual_recall.focus
    manual = select_chat_context(
        instructions=instructions,
        recent_turns=recent,
        current_turn=case.question,
        seed=manual_seed,
        focus=manual_focus,
        candidates=_manual_candidates(
            case,
            [message_id for message_id in manual_ids if message_id != manual_seed.id],
            manual_focus,
            blocked,
        ),
        window=window,
        output_reserve=reserve,
        counter=counter,
        top_k=int(config.get("top_k", 3)),
        threshold=float(config.get("attention_threshold", 0.05)),
        include_seed=True,
        use_utility=False,
    )
    outcomes["manual"] = evaluate_context(
        "manual",
        manual.messages,
        _memory_ids(manual_seed, manual.diagnostics),
        manual.diagnostics.memory_tokens,
    )
    return CaseResult(case=case, outcomes=outcomes)


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> list[CaseResult]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    return [_run_case(case, config, settings, client) for case in cases]


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    cases_with_expected = [result for result in results if result.case.expected_ids]
    for condition in CONDITIONS:
        outcomes = [result.outcomes[condition] for result in results]
        expected_outcomes = [
            result.outcomes[condition] for result in cases_with_expected
        ]
        hit_count = sum(outcome.hit_any for outcome in expected_outcomes)
        full_count = sum(outcome.full_coverage for outcome in expected_outcomes)
        summary[condition] = {
            "cases": len(outcomes),
            "hit_denominator": len(expected_outcomes),
            "hit_count": hit_count,
            "full_count": full_count,
            "hit_rate": round(hit_count / len(expected_outcomes), 4)
            if expected_outcomes
            else 0.0,
            "full_rate": round(full_count / len(expected_outcomes), 4)
            if expected_outcomes
            else 0.0,
            "avg_additional_tokens": round(
                sum(outcome.additional_tokens for outcome in outcomes) / len(outcomes), 2
            ),
            "restricted_violations": sum(
                len(outcome.restricted_violations) for outcome in outcomes
            ),
            "repetitions": sum(outcome.repetition for outcome in outcomes),
            "irrelevant_attachments": sum(
                outcome.irrelevant_attached for outcome in outcomes
            ),
            "avg_answer_score": round(
                sum(outcome.answer_score for outcome in outcomes) / len(outcomes), 4
            ),
        }

    deltas: dict[str, float] = {}
    for baseline in ("history", "lexical", "manual"):
        deltas[baseline] = round(
            (summary["sinapse"]["avg_answer_score"] - summary[baseline]["avg_answer_score"])
            * 100,
            2,
        )
    per_case: list[dict[str, Any]] = []
    for result in results:
        scores = {
            condition: result.outcomes[condition].answer_score
            for condition in CONDITIONS
        }
        better = [
            baseline
            for baseline in ("history", "lexical", "manual")
            if scores["sinapse"] > scores[baseline]
        ]
        worse = [
            baseline
            for baseline in ("history", "lexical", "manual")
            if scores["sinapse"] < scores[baseline]
        ]
        per_case.append(
            {
                "case": result.case.id,
                "kind": result.case.kind,
                "scores": scores,
                "better_than": better,
                "worse_than": worse,
                "selected": {
                    condition: list(result.outcomes[condition].selected_ids)
                    for condition in CONDITIONS
                },
                "additional_tokens": {
                    condition: result.outcomes[condition].additional_tokens
                    for condition in CONDITIONS
                },
                "expected_ids": list(result.case.expected_ids),
                "restricted_violations": {
                    condition: list(result.outcomes[condition].restricted_violations)
                    for condition in CONDITIONS
                },
            }
        )
    return {
        "conditions": summary,
        "sinapse_delta_pp": deltas,
        "per_case": per_case,
        "cases": len(results),
    }


def export(
    results: Sequence[CaseResult], summary: Mapping[str, Any], out_dir: str | Path
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "summary": dict(summary),
        "cases": [result.as_dict() for result in results],
    }
    (out / "results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review: list[dict[str, Any]] = []
    counter = 0
    for result in results:
        for condition in CONDITIONS:
            counter += 1
            outcome = result.outcomes[condition]
            review.append(
                {
                    "review_id": f"R{counter:03d}",
                    "answer": outcome.answer,
                    "rubric": result.case.rubric,
                    "required_facts": list(result.case.required_facts),
                    "forbidden_claims": list(result.case.forbidden_claims),
                    "sensitive_terms": list(result.case.forbidden_terms),
                    "must_acknowledge_missing": result.case.must_acknowledge_missing,
                }
            )
    (out / "review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "BENCHMARK SINAPSE",
        f"casos={summary['cases']} | denominador hit/full={summary['conditions']['sinapse']['hit_denominator']} casos com evidência esperada",
        "",
        "condição         hit (n/den)   full (n/den)  tokens  violações  repet.  irrelev.  resposta",
    ]
    for condition in CONDITIONS:
        item = summary["conditions"][condition]
        hit_label = f"{item['hit_count']}/{item['hit_denominator']}={item['hit_rate']:.2f}"
        full_label = f"{item['full_count']}/{item['hit_denominator']}={item['full_rate']:.2f}"
        lines.append(
            f"{condition:<15} {hit_label:<14} {full_label:<14} "
            f"{item['avg_additional_tokens']:>7.1f} {item['restricted_violations']:>9} "
            f"{item['repetitions']:>7} {item['irrelevant_attachments']:>9} "
            f"{item['avg_answer_score']:>8.2f}"
        )
    lines.append("")
    lines.append("delta Sinapse (pontos percentuais):")
    for baseline, delta in summary["sinapse_delta_pp"].items():
        lines.append(f"  vs {baseline}: {delta:+.2f} pp")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "results": str(out / "results.json"),
        "review": str(out / "review.json"),
        "summary": str(out / "summary.txt"),
    }
