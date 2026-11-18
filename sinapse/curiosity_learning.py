"""Curiosidade aprendida: decide entre encerrar e três buscas extras.

Ações: ``stop``, ``follow_links``, ``other_theme`` e ``correction``.
Cada ação é executada a partir do mesmo resultado inicial e passa pelo mesmo
seletor/orçamento. O rótulo de treino é a ação com maior ganho em evidências
esperadas; empates preferem menos inclusões irrelevantes, menos candidatos
extras e, por fim, ``stop``.

O modelo usa apenas sinais disponíveis antes da busca extra.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attributes import ATTRIBUTE_PRICE, attribute_targets, requested_attribute, supporting_ids
from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages, select_chat_context
from .config import load_config
from .memory import association as association_for, evidence_vector, recall
from .models import Candidate, Message, load_messages
from .sensitivity import SensitivityPolicy
from .text import infer_dimension, normalize_text, tokenize
from .themes import THEME_LEXICONS, analyze_turn

ACTIONS = ("stop", "follow_links", "other_theme", "correction")
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CuriosityCase:
    id: str
    family: str
    messages: tuple[Message, ...]
    conversation_id: str
    recent: tuple[RecentTurn, ...]
    question: str
    expected_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    seed_id: str
    hidden_ids: tuple[str, ...]
    split: str = "train"
    memory_limit: int = 140
    preferences: tuple[str, ...] = ()


def _load(*files: str) -> tuple[Message, ...]:
    messages: list[Message] = []
    for relative in files:
        messages.extend(load_messages(ROOT / relative))
    return tuple(messages)


def _recent(*texts: str, ids: Sequence[str | None] = ()) -> tuple[RecentTurn, ...]:
    if not ids:
        ids = (None,) * len(texts)
    return tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(texts, ids)
    )


def cases() -> tuple[list[CuriosityCase], list[CuriosityCase]]:
    base = "data/agent_conversation.json"
    train = [
        CuriosityCase("tr_follow", "serum", _load(base), "c1", _recent("Falamos do Serum 2 e do bass."), "Por que eu escolhi o preset Serum 2?", ("m2",), ("m1",), "m1", ("m2",)),
        CuriosityCase("tr_follow_ruido", "serum", _load(base, "benchmark/extra_similar.json"), "c1", _recent("Falamos do Serum 2 e do bass."), "Por que eu escolhi o preset Serum 2?", ("m2",), ("m1",), "m1", ("m2",)),
        CuriosityCase("tr_other", "pack", _load(base), "c1", _recent("Falamos do Serum 2 e do pack."), "E aquilo do pack?", ("m9",), ("m7",), "m7", ("m9",), memory_limit=180),
        CuriosityCase("tr_correction", "correcao", _load(base, "benchmark/v2_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m20",), ("m1",), "m1", ("m20",)),
        CuriosityCase("tr_price_link", "preco", _load(base, "benchmark/v2_price_reply.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m1",), "m1", ("m20",), memory_limit=110),
        CuriosityCase("tr_stop", "preco", _load(base), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", (), (), "m1", ()),
        CuriosityCase("tr_sensitive", "sensibilidade", _load(base), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), (), "m1", ("m2",), preferences=("Não quero falar sobre a licença do Serum 2.",)),
    ]
    evaluation = [
        CuriosityCase("ev_follow", "bicicleta", _load(base), "c2", _recent("Falamos de bicicleta."), "Por que comprei a bicicleta?", ("p2",), ("p1",), "p1", ("p2",), split="eval"),
        CuriosityCase("ev_other", "pack", _load(base), "c1", _recent("Falamos do Serum 2 e do pack."), "E aquilo do pack?", ("m9",), ("m7",), "m7", ("m9",), split="eval", memory_limit=180),
        CuriosityCase("ev_correction", "correcao", _load(base, "benchmark/extra_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m10",), ("m1",), "m1", ("m10",), split="eval"),
        CuriosityCase("ev_stop", "preco", _load(base), "c1", _recent("Falamos do Piano V."), "Quanto custou o Piano V?", (), (), "m5", (), split="eval"),
        CuriosityCase("ev_price_link", "preco", _load(base, "benchmark/v2_price_reply.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m1",), "m1", ("m20",), split="eval"),
        CuriosityCase("ev_sensitive", "sensibilidade", _load(base), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), (), "m1", ("m2",), split="eval", preferences=("Não quero falar sobre a licença do Serum 2.",)),
    ]
    return train, evaluation


def _id_in_messages(message_id: str, messages: Sequence[Mapping[str, str]]) -> bool:
    context = "\n".join(str(message["content"]) for message in messages[:-1])
    pattern = rf"(?<![a-z0-9_])id={re.escape(message_id)}(?!\d)"
    return bool(re.search(pattern, context)) or f"[{message_id}]" in context


def _memory_ids(seed: Message, diagnostics) -> list[str]:
    ids = [seed.id] if diagnostics.seed_included else []
    ids.extend(diagnostics.selected_ids)
    return list(dict.fromkeys(ids))


def _extra_candidate(
    message: Message,
    seed: Message,
    recall_result,
    config: Mapping[str, Any],
) -> Candidate:
    from .models import DIMENSIONS

    evidence = {dimension: None for dimension in DIMENSIONS}
    evidence[recall_result.focus] = 1.0
    return Candidate(
        message=message,
        distance=0.0,
        association=0.9,
        association_content=0.9,
        association_link=0.0,
        association_source="extra_search",
        evidence=evidence,
        attention=0.9,
        within_radius=True,
    )


@dataclass
class InitialState:
    case: CuriosityCase
    recall_result: Any
    seed: Message
    base_messages: tuple[dict[str, str], ...]
    base_tokens: int
    window: int
    reserve: int
    candidates: list[Candidate]
    blocked: set[str]
    counter: EstimatedChatCounter
    attribute: str | None
    targets: frozenset[str]


def _initial_state(case: CuriosityCase, config: Mapping[str, Any]) -> InitialState:
    policy = SensitivityPolicy()
    for preference in case.preferences:
        policy.declare(case.conversation_id, preference, 0)
    blocked = set(
        policy.evaluate(case.conversation_id, case.question, case.messages, 0).blocked_ids
    )
    recall_result = recall(
        case.messages,
        case.question,
        [turn.text for turn in case.recent],
        config,
        seed_id=case.seed_id,
        conversation_id=case.conversation_id,
        excluded_ids=tuple(blocked),
    )
    seed = recall_result.seed
    candidates = [
        candidate
        for candidate in recall_result.candidates
        if candidate.message.id not in case.hidden_ids
        and candidate.message.id not in blocked
    ]
    base = build_base_messages(
        str(config.get("instructions", "")), case.recent, case.question
    )
    counter = EstimatedChatCounter(SimulatedCounter())
    base_tokens = counter.count_messages(base)
    return InitialState(
        case=case,
        recall_result=recall_result,
        seed=seed,
        base_messages=base,
        base_tokens=base_tokens,
        window=base_tokens + 4096 + case.memory_limit,
        reserve=4096,
        candidates=candidates,
        blocked=blocked,
        counter=counter,
        attribute=requested_attribute(case.question),
        targets=attribute_targets(case.question),
    )


def _action_messages(
    action: str,
    state: InitialState,
) -> list[Message]:
    case = state.case
    if action == "stop":
        return []
    by_id = {message.id: message for message in case.messages}
    if action == "follow_links":
        ids: set[str] = set()
        for message in [state.seed, *[item.message for item in state.candidates[:3]]]:
            if message.reply_to:
                ids.add(message.reply_to)
            ids.update(
                other.id for other in case.messages if other.reply_to == message.id
            )
        return [by_id[message_id] for message_id in ids if message_id in by_id]
    if action == "correction":
        terms = ("troquei", "deixei", "corrigi", "na verdade", "agora uso", "mudei")
        return [
            message
            for message in case.messages
            if any(term in normalize_text(message.text) for term in terms)
        ]
    # other_theme: termos do turno/histórico fora da semente, sem correções.
    tokens = set(tokenize(case.question))
    for turn in case.recent:
        tokens.update(tokenize(turn.text))
    other = tokens
    correction_terms = ("troquei", "deixei", "corrigi", "na verdade", "agora uso")
    return [
        message
        for message in case.messages
        if message.id != state.seed.id
        and set(tokenize(message.text)) & other
        and not any(term in normalize_text(message.text) for term in correction_terms)
    ]


def _run_action(
    action: str,
    state: InitialState,
    config: Mapping[str, Any],
    min_gain: float = 0.05,
    min_novelty: float = 0.55,
    redundancy_weight: float = 0.6,
) -> dict[str, Any]:
    started = time.perf_counter()
    extras: list[Message] = []
    existing = {candidate.message.id for candidate in state.candidates}
    for message in _action_messages(action, state):
        if message.id in existing or message.id in state.blocked:
            continue
        if state.attribute == ATTRIBUTE_PRICE and not message_supports_price(
            message, state
        ):
            continue
        extras.append(message)
        existing.add(message.id)
    candidates = list(state.candidates) + [
        _extra_candidate(message, state.seed, state.recall_result, config)
        for message in extras
    ]
    result = select_chat_context(
        instructions=str(config.get("instructions", "")),
        recent_turns=state.case.recent,
        current_turn=state.case.question,
        seed=state.seed,
        focus=state.recall_result.focus,
        candidates=candidates,
        window=state.window,
        output_reserve=state.reserve,
        counter=state.counter,
        top_k=int(config.get("top_k", 3)),
        threshold=float(config.get("attention_threshold", 0.05)),
        include_seed=True,
        use_utility=False,
    )
    selected = _memory_ids(state.seed, result.diagnostics)
    full = sum(
        1 for message_id in state.case.expected_ids
        if _id_in_messages(message_id, result.messages)
    )
    irrelevant = [
        message_id
        for message_id in selected
        if message_id not in state.case.expected_ids
        and message_id not in state.case.dependency_ids
    ]
    return {
        "action": action,
        "full": full,
        "selected": selected,
        "irrelevant": irrelevant,
        "extras": len(extras),
        "candidate_ids": [candidate.message.id for candidate in candidates],
        "tokens": result.diagnostics.used_tokens,
        "memory_tokens": result.diagnostics.memory_tokens,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "messages": result.messages,
    }


def message_supports_price(message: Message, state: InitialState) -> bool:
    from .attributes import message_supports_attribute

    by_id = {item.id: item for item in state.case.messages}
    return message_supports_attribute(
        message, ATTRIBUTE_PRICE, state.targets, by_id
    )


def _available_actions(state: InitialState) -> list[str]:
    available = ["stop"]
    links = any(message.reply_to for message in state.case.messages) or any(
        item.message.reply_to for item in state.candidates[:3]
    )
    if links:
        available.append("follow_links")
    themes = set(state.recall_result.weights)
    if len(themes) >= 2 or len(state.case.expected_ids) > 0:
        available.append("other_theme")
    if any(
        term in normalize_text(message.text)
        for message in state.case.messages
        for term in ("troquei", "deixei", "corrigi", "na verdade", "agora uso")
    ):
        available.append("correction")
    return available


def _features(state: InitialState) -> list[float]:
    theme_report = analyze_turn(
        state.case.question,
        [turn.text for turn in state.case.recent],
        {},
        [message.text for message in state.case.messages[:0]],
    )
    normalized = normalize_text(state.case.question)
    pronoun = any(
        token in normalized.split()
        for token in ("ele", "ela", "isso", "isto", "aquilo", "esse", "essa")
    )
    short = len(tokenize(state.case.question)) <= 3
    ambiguous = 1.0 if (len(theme_report.themes) >= 2 or (pronoun and short)) else 0.0
    initial_evidence = any(
        candidate.evidence.get(state.recall_result.focus) is not None
        and candidate.attention > 0.05
        for candidate in state.candidates
    )
    attribute_available = 0.0
    if state.attribute:
        by_id = {message.id: message for message in state.case.messages}
        support = supporting_ids(
            state.case.messages,
            state.attribute,
            state.case.conversation_id,
            state.targets,
            by_id,
        )
        attribute_available = 1.0 if support else 0.0
        initial_support = state.seed.id in support or any(
            candidate.message.id in support for candidate in state.candidates
        )
        missing = 0.0 if initial_support else 1.0
    else:
        missing = 0.0 if initial_evidence else 1.0
    conflict = 1.0 if any(
        term in normalize_text(message.text)
        for message in state.case.messages
        for term in ("troquei", "deixei", "corrigi", "na verdade", "agora uso")
    ) else 0.0
    links = 1.0 if any(
        message.reply_to == state.seed.id for message in state.case.messages
    ) or any(
        message.reply_to
        for message in [state.seed, *[c.message for c in state.candidates[:5]]]
    ) else 0.0
    return [
        ambiguous,
        missing,
        conflict,
        links,
        min(1.0, len(state.candidates) / 50.0),
        attribute_available,
        1.0 if state.attribute else 0.0,
        1.0 if state.case.preferences else 0.0,
    ]


def _softmax(scores: Sequence[float]) -> list[float]:
    peak = max(scores)
    exp = [math.exp(score - peak) for score in scores]
    total = sum(exp)
    return [value / total for value in exp]


def train_policy(rows: Sequence[tuple[list[float], int]], epochs: int = 500, lr: float = 0.5) -> dict[str, Any]:
    feature_count = len(rows[0][0])
    weights = [[0.0] * feature_count for _ in ACTIONS]
    biases = [0.0] * len(ACTIONS)
    for _ in range(epochs):
        gradients = [[0.0] * feature_count for _ in ACTIONS]
        bias_gradients = [0.0] * len(ACTIONS)
        for features, label in rows:
            scores = [
                sum(weight * value for weight, value in zip(weights[index], features))
                + biases[index]
                for index in range(len(ACTIONS))
            ]
            probabilities = _softmax(scores)
            for index in range(len(ACTIONS)):
                error = probabilities[index] - (1.0 if index == label else 0.0)
                for feature_index, value in enumerate(features):
                    gradients[index][feature_index] += error * value
                bias_gradients[index] += error
        for index in range(len(ACTIONS)):
            for feature_index in range(feature_count):
                weights[index][feature_index] -= lr * gradients[index][feature_index] / len(rows)
            biases[index] -= lr * bias_gradients[index] / len(rows)
    return {"weights": weights, "biases": biases}


def predict(model: Mapping[str, Any], features: Sequence[float]) -> str:
    scores = [
        sum(weight * value for weight, value in zip(model["weights"][index], features))
        + model["biases"][index]
        for index in range(len(ACTIONS))
    ]
    return ACTIONS[max(range(len(ACTIONS)), key=lambda index: scores[index])]


def fixed_action(state: InitialState) -> str:
    features = _features(state)
    ambiguous, missing, conflict, links, _, _, _, _ = features
    if ambiguous:
        return "other_theme"
    if missing and links:
        return "follow_links"
    if conflict:
        return "correction"
    return "stop"


def run_experiment(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    train_cases, eval_cases = cases()
    training_rows: list[tuple[list[float], int]] = []
    training_details = []
    for case in train_cases:
        state = _initial_state(case, config)
        results = {action: _run_action(action, state, config) for action in _available_actions(state)}
        stop = results["stop"]
        best = "stop"
        best_key = (0, -len(stop["irrelevant"]), -stop["extras"])
        for action, result in results.items():
            key = (
                result["full"] - stop["full"],
                -len(result["irrelevant"]),
                -result["extras"],
            )
            if key > best_key:
                best_key = key
                best = action
        features = _features(state)
        training_rows.append((features, ACTIONS.index(best)))
        training_details.append(
            {
                "case": case.id,
                "best_action": best,
                "actions": {
                    action: {
                        "full": result["full"],
                        "irrelevant": result["irrelevant"],
                        "extras": result["extras"],
                    }
                    for action, result in results.items()
                },
            }
        )
    model = train_policy(training_rows)

    evaluation = []
    for case in eval_cases:
        state = _initial_state(case, config)
        stop = _run_action("stop", state, config)
        fixed = _run_action(fixed_action(state), state, config)
        learned_action = predict(model, _features(state))
        learned = _run_action(learned_action, state, config)
        evaluation.append(
            {
                "case": case.id,
                "family": case.family,
                "features": dict(
                    zip(
                        ("ambiguidade", "faltante", "conflito", "vinculos", "candidatos", "atributo_disponivel", "atributo_pedido", "restricao"),
                        _features(state),
                    )
                ),
                "fixed_action": fixed["action"],
                "learned_action": learned_action,
                "stop": stop,
                "fixed": fixed,
                "learned": learned,
                "improved": learned["full"] > fixed["full"],
                "worsened": learned["full"] < fixed["full"],
            }
        )

    def aggregate(key: str) -> dict[str, Any]:
        items = [entry[key] for entry in evaluation]
        no_gain = 0
        for entry in evaluation:
            item = entry[key]
            if item["action"] != "stop" and item["full"] <= entry["stop"]["full"]:
                no_gain += 1
        return {
            "full": sum(item["full"] for item in items),
            "no_gain_searches": no_gain,
            "irrelevant": sum(len(item["irrelevant"]) for item in items),
            "extras": sum(item["extras"] for item in items),
            "latency_median_ms": sorted(item["elapsed_ms"] for item in items)[len(items) // 2],
            "avg_tokens": round(sum(item["tokens"] for item in items) / len(items), 1),
        }

    return {
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "training_rows": len(training_rows),
        "training_details": training_details,
        "model": model,
        "evaluation": evaluation,
        "off": aggregate("stop"),
        "fixed_rules": aggregate("fixed"),
        "learned": aggregate("learned"),
    }


def entry_stop_full(evaluation: Sequence[Mapping[str, Any]], item: Mapping[str, Any]) -> int:
    for entry in evaluation:
        if entry["case"] == item.get("case"):
            return entry["stop"]["full"]
    return 0
