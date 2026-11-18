"""Experimento local de redução de contexto.

Compara o seletor atual (mais_treino congelado) com uma variante que reavalia
o que cada item acrescenta ao histórico recente e às evidências já escolhidas.
A busca inicial, a ação de curiosidade, os candidatos, a sensibilidade, o
renderizador e o orçamento são os mesmos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import (
    RecentTurn,
    SimulatedCounter,
    document_signature,
    is_duplicate,
    render_evidence_block,
    render_seed_block,
    wrap_memory,
)
from .chat import EstimatedChatCounter, with_memory
from .config import load_config
from .curiosity_learning import (
    ACTIONS,
    CuriosityCase,
    _available_actions,
    _features,
    _initial_state,
    _load,
    _recent,
    _run_action,
    cases as base_cases,
    train_policy,
)
from .curiosity_round2 import _aggregate, _expected_and_context, probabilities
from .curiosity_round3 import _eval_cases as round3_eval_cases, _training_cases
from .memory import link_between, message_tokens, requires_seed
from .models import Message
from .text import normalize_text, tokenize
from .utility import estimate_utility

BASE = "data/agent_conversation.json"
CORRECTION_TERMS = ("troquei", "deixei", "corrigi", "na verdade", "agora uso", "mudei")


def _extra_eval_cases() -> list[CuriosityCase]:
    return [
        CuriosityCase(
            "ev_self_sufficient",
            "autossuficiente",
            _load(BASE, "benchmark/v2_price_message.json"),
            "c1",
            _recent("Falamos do Serum 2."),
            "Quanto custou o Serum 2?",
            ("m20",),
            ("m20",),
            "m20",
            (),
            split="eval",
            memory_limit=90,
        ),
        CuriosityCase(
            "ev_two_complementary",
            "duas_evidencias",
            _load(BASE, "benchmark/extra_two.json"),
            "c1",
            _recent("Falamos do Serum 2 e do bass."),
            "Por que eu escolhi o preset Serum 2?",
            ("m2", "m10"),
            ("m1",),
            "m1",
            (),
            split="eval",
            memory_limit=180,
        ),
    ]


def _train_more_model(config: Mapping[str, Any]):
    rows = []
    for case in _training_cases():
        state = _initial_state(case, config)
        results = {
            action: _run_action(action, state, config)
            for action in _available_actions(state)
        }
        stop = results["stop"]
        best = "stop"
        best_key = (0, -len(stop["irrelevant"]), -stop["extras"])
        for action, value in results.items():
            key = (
                value["full"] - stop["full"],
                -len(value["irrelevant"]),
                -value["extras"],
            )
            if key > best_key:
                best_key = key
                best = action
        rows.append((_features(state), ACTIONS.index(best)))
    return train_policy(rows)


def _history_resolved(state, seed: Message) -> bool:
    for turn in state.case.recent:
        message = next(
            (item for item in state.case.messages if item.text == turn.text),
            None,
        )
        if turn.id:
            message = next(
                (item for item in state.case.messages if item.id == turn.id), message
            )
        if message is None or message.id == seed.id:
            continue
        if any(message.values(dimension) for dimension in ("o_que", "por_que", "como", "para_que")):
            return True
    return False


def _query_overlap(message: Message, question: str) -> float:
    tokens = message_tokens(message)
    query = tokenize(question)
    if not tokens or not query:
        return 0.0
    return len(tokens & query) / len(tokens)


def _reduction(
    state,
    current: Mapping[str, Any],
    config: Mapping[str, Any],
    min_gain: float = 0.0,
    min_novelty: float = 0.55,
    redundancy_weight: float = 0.6,
) -> dict[str, Any]:
    seed = state.seed
    current_ids = list(current["selected"])
    seed_was_included = seed.id in current_ids
    evidence_ids = [message_id for message_id in current_ids if message_id != seed.id]
    by_id = {message.id: message for message in state.case.messages}
    recent_texts = [turn.text for turn in state.case.recent]
    kept: list[Message] = []
    removed: list[str] = []
    keep_seed = False

    history_done = _history_resolved(state, seed)
    if history_done:
        kept = []
        removed = list(current_ids)
    else:
        for message_id in evidence_ids:
            message = by_id.get(message_id)
            if message is None:
                removed.append(message_id)
                continue
            if any(normalize_text(message.text) == normalize_text(text) for text in recent_texts):
                removed.append(message_id)
                continue
            if kept and any(is_duplicate(message, other) for other in kept):
                removed.append(message_id)
                continue
            utility = estimate_utility(
                message,
                1.0,
                1.0,
                recent_texts,
                kept,
                redundancy_weight=redundancy_weight,
            )
            correction = any(
                term in normalize_text(message.text) for term in CORRECTION_TERMS
            )
            overlap = _query_overlap(message, state.case.question)
            complementary = (
                overlap >= 0.2
                and bool(kept)
                and any(
                    document_signature(message) != document_signature(other)
                    for other in kept
                )
            )
            linked = link_between(message, seed)
            if (
                not correction
                and not linked
                and overlap < 0.2
                and not complementary
            ):
                removed.append(message_id)
                continue
            if (
                (utility.novelty < min_novelty or utility.gain < min_gain)
                and overlap < 0.25
                and not correction
                and not complementary
            ):
                removed.append(message_id)
                continue
            kept.append(message)

        correction_query = any(
            term in normalize_text(state.case.question)
            for term in CORRECTION_TERMS + ("agora",)
        )
        if correction_query:
            corrections = [
                message
                for message in kept
                if any(term in normalize_text(message.text) for term in CORRECTION_TERMS)
            ]
            if corrections:
                for message in kept:
                    if message not in corrections:
                        removed.append(message.id)
                kept = corrections

        keep_seed = seed_was_included
        if keep_seed:
            needs_seed = any(requires_seed(message, seed) for message in kept)
            self_sufficient = bool(kept) and all(
                not requires_seed(message, seed) for message in kept
            )
            seed_is_answer = (
                state.attribute == "preco"
                and seed.values("o_que")
                and _query_overlap(seed, state.case.question) > 0
            )
            if not kept:
                keep_seed = seed_is_answer
            elif needs_seed:
                keep_seed = True
            elif self_sufficient:
                keep_seed = False
        if not keep_seed and seed_was_included:
            removed.append(seed.id)
    selected_ids = ([seed.id] if keep_seed else []) + [message.id for message in kept]
    blocks = []
    if keep_seed:
        blocks.append(render_seed_block(seed))
    for rank, message in enumerate(kept, start=1):
        blocks.append(render_evidence_block(message, rank))
    memory_text = wrap_memory(blocks)
    messages = with_memory(state.base_messages, memory_text)
    counter = EstimatedChatCounter(SimulatedCounter())
    used = counter.count_messages(messages)
    return {
        "selected": selected_ids,
        "removed": removed,
        "messages": messages,
        "memory_tokens": used - state.base_tokens,
        "tokens": used,
    }


def run_context_reduction(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    model = _train_more_model(config)
    cases = list(round3_eval_cases()) + _extra_eval_cases()
    entries = []
    for case in cases:
        state = _initial_state(case, config)
        action = max(
            ACTIONS, key=lambda item: probabilities(model, _features(state))[item]
        )
        current = _run_action(action, state, config)
        reduced = _reduction(state, current, config)
        current_expected, current_context, current_functionless = _expected_and_context(
            case, current
        )
        reduced_expected, reduced_context, reduced_functionless = _expected_and_context(
            case,
            {
                "selected": reduced["selected"],
                "messages": reduced["messages"],
            },
        )
        by_id = {message.id: message for message in case.messages}
        seed_kept = state.seed.id in reduced["selected"]
        necessary_context_loss = (
            not seed_kept
            and any(
                requires_seed(by_id[message_id], state.seed)
                for message_id in reduced["selected"]
                if message_id in by_id and message_id != state.seed.id
            )
        )
        entries.append(
            {
                "case": case.id,
                "family": case.family,
                "action": action,
                "current": {
                    "selected": current["selected"],
                    "expected": current_expected,
                    "context": current_context,
                    "functionless": current_functionless,
                    "memory_tokens": current["memory_tokens"],
                    "tokens": current["tokens"],
                },
                "reduced": {
                    "selected": reduced["selected"],
                    "expected": reduced_expected,
                    "context": reduced_context,
                    "functionless": reduced_functionless,
                    "memory_tokens": reduced["memory_tokens"],
                    "tokens": reduced["tokens"],
                    "removed": reduced["removed"],
                    "necessary_context_loss": necessary_context_loss,
                },
            }
        )

    def aggregate(side: str) -> dict[str, Any]:
        items = [entry[side] for entry in entries]
        return {
            "expected": sum(len(item["expected"]) for item in items),
            "context": sum(len(item["context"]) for item in items),
            "functionless": sum(len(item["functionless"]) for item in items),
            "memory_tokens": sum(item["memory_tokens"] for item in items),
            "tokens": sum(item["tokens"] for item in items),
            "cases": len(items),
        }

    tradeoffs = [
        entry
        for entry in entries
        if len(entry["reduced"]["expected"]) < len(entry["current"]["expected"])
        or entry["reduced"]["necessary_context_loss"]
    ]
    return {
        "cases": len(entries),
        "current": aggregate("current"),
        "reduced": aggregate("reduced"),
        "entries": entries,
        "tradeoffs": [entry["case"] for entry in tradeoffs],
    }
