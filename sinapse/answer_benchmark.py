"""Benchmark de qualidade de resposta por condição de memória.

Quatro condições:
- history: apenas histórico recente;
- distractors: mensagens tematicamente próximas que não respondem;
- sinapse: seleção da Sinapse;
- manual: evidência correta + dependências.

Mesmo modelo, instruções, pergunta, histórico, parâmetros de geração, teto de
memória e renderizador. Mede evidência enviada, evidência usada, custo e nota.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark import (
    BenchmarkCase,
    SimulatedAnswerClient,
    _prepare,
    _present_ids,
    load_cases,
    score_answer,
)
from .budget import SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages, select_chat_context
from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekClient,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .memory import recall
from .models import Candidate, Message
from .sensitivity import SensitivityPolicy
from .text import cosine_overlap, infer_dimension, normalize_text
from .whiteboard import MemoryAgent

CONDITIONS = ("history", "distractors", "sinapse", "manual")


@dataclass
class AnswerOutcome:
    condition: str
    messages: tuple[dict[str, str], ...]
    selected_ids: tuple[str, ...]
    evidence_sent: bool
    evidence_used: bool
    answer: str
    score: float
    checks: Mapping[str, Any]
    additional_tokens: int
    used_tokens: int
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "selected_ids": list(self.selected_ids),
            "evidence_sent": self.evidence_sent,
            "evidence_used": self.evidence_used,
            "answer": self.answer,
            "score": self.score,
            "checks": dict(self.checks),
            "additional_tokens": self.additional_tokens,
            "used_tokens": self.used_tokens,
            "error": self.error,
        }


@dataclass
class AnswerCaseResult:
    case: BenchmarkCase
    outcomes: dict[str, AnswerOutcome]

    def as_dict(self) -> dict:
        return {
            "case": self.case.id,
            "kind": self.case.kind,
            "outcomes": {
                name: outcome.as_dict() for name, outcome in self.outcomes.items()
            },
        }


def _memory_ids(seed: Message, diagnostics) -> tuple[str, ...]:
    ids: list[str] = []
    if diagnostics.seed_included:
        ids.append(seed.id)
    ids.extend(diagnostics.selected_ids)
    return tuple(dict.fromkeys(ids))


def _candidate(message: Message, focus: str) -> Candidate:
    evidence = {dimension: None for dimension in ("o_que", "quem", "onde", "quando", "como", "por_que", "para_que")}
    evidence[focus] = 1.0
    return Candidate(
        message=message,
        distance=0.0,
        association=1.0,
        association_content=1.0,
        association_link=0.0,
        association_source="benchmark",
        evidence=evidence,
        attention=1.0,
        within_radius=True,
    )


def _distractor_messages(
    case: BenchmarkCase,
    blocked: set[str],
    policy: SensitivityPolicy,
) -> list[Message]:
    by_id = {message.id: message for message in case.messages}
    excluded = set(case.expected_ids) | set(case.dependency_ids) | blocked
    candidates = [
        by_id[message_id]
        for message_id in case.distractor_ids
        if message_id in by_id and message_id not in excluded
    ]
    if candidates:
        return candidates
    # Fallback: mensagens com maior similaridade lexical que não são a resposta.
    from .memory import message_tokens
    from .text import tokenize

    query_tokens = tokenize(case.question)
    scored = []
    for message in case.messages:
        if message.id in excluded or message.conversation_id != case.conversation_id:
            continue
        score = cosine_overlap(message_tokens(message), query_tokens)
        if score > 0:
            scored.append((score, message))
    scored.sort(key=lambda item: (-item[0], item[1].position))
    return [message for _, message in scored[:3]]


def _run_condition(
    condition: str,
    case: BenchmarkCase,
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
    policy: SensitivityPolicy,
    recent,
    instructions: str,
    base_tokens: int,
    window: int,
    reserve: int,
    client: ChatTransport | None,
) -> AnswerOutcome:
    by_id = {message.id: message for message in case.messages}
    blocked = set(
        policy.evaluate(case.conversation_id, case.question, case.messages, 0).blocked_ids
    )
    focus, _ = infer_dimension(case.question)
    selected_ids: tuple[str, ...] = ()
    messages: tuple[dict[str, str], ...]
    counter = EstimatedChatCounter(SimulatedCounter())

    if condition == "history":
        messages = build_base_messages(instructions, recent, case.question)
        additional = 0
    elif condition == "distractors":
        distractors = _distractor_messages(case, blocked, policy)
        if not distractors:
            messages = build_base_messages(instructions, recent, case.question)
            additional = 0
        else:
            seed = distractors[0]
            candidates = [_candidate(message, focus) for message in distractors[1:]]
            result = select_chat_context(
                instructions=instructions,
                recent_turns=recent,
                current_turn=case.question,
                seed=seed,
                focus=focus,
                candidates=candidates,
                window=window,
                output_reserve=reserve,
                counter=counter,
                top_k=int(config.get("top_k", 3)),
                threshold=float(config.get("attention_threshold", 0.05)),
                include_seed=True,
                use_utility=False,
            )
            messages = result.messages
            selected_ids = _memory_ids(seed, result.diagnostics)
            additional = result.diagnostics.memory_tokens
    elif condition == "sinapse":
        agent = MemoryAgent(case.messages, config)
        for preference in case.preferences:
            agent.declare_preference(case.conversation_id, preference, 0)
        for summary in case.summaries:
            agent.add_summary(
                summary.conversation_id, summary.text, summary.source_ids, summary.id
            )
        turn = agent.before_response(
            case.conversation_id,
            case.question,
            recent,
            window=window,
            output_reserve=reserve,
        )
        messages = turn.messages
        selected_ids = _memory_ids(turn.seed, turn.budget.diagnostics)
        additional = turn.budget.diagnostics.memory_tokens
    else:
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
        if not case.expected_ids:
            messages = build_base_messages(instructions, recent, case.question)
            additional = 0
        else:
            seed_id = case.dependency_ids[0] if case.dependency_ids else None
            recall_result = recall(
                case.messages,
                case.question,
                [turn.text for turn in recent],
                config,
                conversation_id=case.conversation_id,
                seed_id=seed_id if seed_id not in blocked else None,
                excluded_ids=tuple(blocked),
            )
            seed = (
                by_id[seed_id]
                if seed_id in by_id and seed_id not in blocked
                else recall_result.seed
            )
            candidates = [
                _candidate(by_id[message_id], recall_result.focus)
                for message_id in manual_ids
                if message_id in by_id and message_id != seed.id and message_id not in blocked
            ]
            result = select_chat_context(
                instructions=instructions,
                recent_turns=recent,
                current_turn=case.question,
                seed=seed,
                focus=recall_result.focus,
                candidates=candidates,
                window=window,
                output_reserve=reserve,
                counter=counter,
                top_k=int(config.get("top_k", 3)),
                threshold=float(config.get("attention_threshold", 0.05)),
                include_seed=True,
                use_utility=False,
            )
            messages = result.messages
            selected_ids = _memory_ids(seed, result.diagnostics)
            additional = result.diagnostics.memory_tokens

    present = set(_present_ids(case, messages))
    evidence_sent = set(case.expected_ids) <= present
    answer = ""
    error = None
    if client is None:
        answer_client: ChatTransport = SimulatedAnswerClient(case, messages)
    else:
        answer_client = client
    try:
        response: ChatCompletionResponse = answer_client.complete(
            build_request_body(messages, settings), timeout=settings.timeout
        )
        answer = response.content
    except Exception as exc:
        error = str(exc)
    score, checks = score_answer(case, answer)
    if case.required_facts:
        evidence_used = all(
            normalize_text(fact) in normalize_text(answer)
            for fact in case.required_facts
        )
    elif case.must_acknowledge_missing:
        evidence_used = any(
            marker in normalize_text(answer)
            for marker in (
                "nao tenho",
                "nao sei",
                "sem informacao",
                "nao consta",
                "nao disponivel",
            )
        )
    else:
        evidence_used = True
    return AnswerOutcome(
        condition=condition,
        messages=messages,
        selected_ids=selected_ids,
        evidence_sent=evidence_sent,
        evidence_used=evidence_used,
        answer=answer,
        score=score,
        checks=checks,
        additional_tokens=int(additional),
        used_tokens=counter.count_messages(messages),
        error=error,
    )


def run_answer_benchmark(
    cases: Sequence[BenchmarkCase],
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> list[AnswerCaseResult]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    results: list[AnswerCaseResult] = []
    for case in cases:
        policy, recent, instructions, base_tokens, window, reserve = _prepare(
            case, config, settings
        )
        outcomes: dict[str, AnswerOutcome] = {}
        for condition in CONDITIONS:
            outcomes[condition] = _run_condition(
                condition,
                case,
                config,
                settings,
                policy,
                recent,
                instructions,
                base_tokens,
                window,
                reserve,
                client,
            )
        results.append(AnswerCaseResult(case=case, outcomes=outcomes))
    return results


def summarize_answers(results: Sequence[AnswerCaseResult]) -> dict:
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        outcomes = [result.outcomes[condition] for result in results]
        summary[condition] = {
            "cases": len(outcomes),
            "evidence_sent": sum(outcome.evidence_sent for outcome in outcomes),
            "evidence_used": sum(outcome.evidence_used for outcome in outcomes),
            "avg_score": round(
                sum(outcome.score for outcome in outcomes) / len(outcomes), 4
            ),
            "avg_additional_tokens": round(
                sum(outcome.additional_tokens for outcome in outcomes) / len(outcomes), 2
            ),
            "total_tokens": sum(outcome.used_tokens for outcome in outcomes),
        }

    changed = 0
    better_history = 0
    worse_history = 0
    distractors_worse = 0
    correct_but_error: list[str] = []
    improved: list[str] = []
    worsened: list[str] = []
    for result in results:
        scores = {condition: result.outcomes[condition].score for condition in CONDITIONS}
        if len(set(scores.values())) > 1:
            changed += 1
        if scores["sinapse"] > scores["history"]:
            better_history += 1
            improved.append(result.case.id)
        elif scores["sinapse"] < scores["history"]:
            worse_history += 1
        if scores["distractors"] < scores["history"]:
            distractors_worse += 1
            worsened.append(result.case.id)
        outcome = result.outcomes["sinapse"]
        if outcome.evidence_sent and outcome.score < 1.0:
            correct_but_error.append(result.case.id)
    summary["comparisons"] = {
        "changed_cases": changed,
        "sinapse_better_than_history": better_history,
        "sinapse_worse_than_history": worse_history,
        "distractors_worse_than_history": distractors_worse,
        "improved": improved,
        "worsened_by_distractors": worsened,
        "correct_evidence_but_error": correct_but_error,
    }
    return summary


def export_answers(
    results: Sequence[AnswerCaseResult], summary: Mapping[str, Any], out_dir: str | Path
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer_results.json").write_text(
        json.dumps(
            {
                "summary": dict(summary),
                "cases": [result.as_dict() for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    review = []
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
                }
            )
    (out / "answer_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["QUALIDADE DE RESPOSTA POR CONDIÇÃO", f"casos={len(results)}", ""]
    for condition in CONDITIONS:
        item = summary[condition]
        lines.append(
            f"{condition:<12} evid_enviada={item['evidence_sent']}/{item['cases']} "
            f"evid_usada={item['evidence_used']}/{item['cases']} "
            f"nota={item['avg_score']:.2f} tokens={item['avg_additional_tokens']:.1f}"
        )
    comp = summary["comparisons"]
    lines.append("")
    lines.append(f"casos em que trocar a memória mudou a nota: {comp['changed_cases']}")
    lines.append(
        f"Sinapse melhor que histórico: {comp['sinapse_better_than_history']} | "
        f"pior: {comp['sinapse_worse_than_history']} | "
        f"distratores piores que histórico: {comp['distractors_worse_than_history']}"
    )
    (out / "answer_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "results": str(out / "answer_results.json"),
        "review": str(out / "answer_review.json"),
        "summary": str(out / "answer_summary.txt"),
    }
