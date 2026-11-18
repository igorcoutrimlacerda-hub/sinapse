"""Experimento de aprendizado supervisionado dos pesos de relevância.

Modelo linear/logístico sobre sinais existentes: evidência na dimensão pedida,
associação com o acontecimento, vínculo de resposta, novidade, distância
inversa e suporte ao atributo pedido. Treino e avaliação são separados por
família de cenário. Pesos são congelados antes da avaliação.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attributes import (
    ATTRIBUTE_PRICE,
    attribute_targets,
    message_supports_attribute,
    requested_attribute,
    supporting_ids as attribute_supporting_ids,
)
from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages, select_chat_context
from .config import load_config
from .deepseek import load_deepseek_settings
from .memory import link_between, recall
from .models import Message, load_messages
from .sensitivity import SensitivityPolicy
from .text import normalize_text, tokenize
from .utility import estimate_utility

FEATURES = (
    "evidencia",
    "associacao",
    "vinculo",
    "novidade",
    "distancia_inversa",
    "atributo",
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "agent_conversation.json"


@dataclass(frozen=True)
class LearningCase:
    id: str
    family: str
    messages: tuple[Message, ...]
    conversation_id: str
    recent: tuple[RecentTurn, ...]
    question: str
    expected_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    memory_limit: int = 140
    split: str = "train"


def _load(*files: str) -> tuple[Message, ...]:
    messages: list[Message] = []
    for relative in files:
        messages.extend(load_messages(ROOT / relative))
    ids = [message.id for message in messages]
    duplicated = sorted({message_id for message_id in ids if ids.count(message_id) > 1})
    if duplicated:
        raise ValueError(f"IDs duplicados: {duplicated}")
    return tuple(messages)


def _recent(*texts: str) -> tuple[RecentTurn, ...]:
    return tuple(RecentTurn(text=text) for text in texts)


def cases() -> tuple[list[LearningCase], list[LearningCase]]:
    train = [
        LearningCase("train_motivo", "serum", _load("data/agent_conversation.json"), "c1", _recent("Falamos do Serum 2 e do bass."), "Por que eu escolhi o preset Serum 2?", ("m2",), ("m1",)),
        LearningCase("train_motivo_ruido", "serum", _load("data/agent_conversation.json", "benchmark/extra_similar.json"), "c1", _recent("Falamos do Serum 2 e do bass."), "Por que eu escolhi o preset Serum 2?", ("m2",), ("m1",)),
        LearningCase("train_procedimento", "procedimento", _load("data/agent_conversation.json"), "c1", _recent("Falamos do Serum 2 e do bass."), "Como eu testei o bass?", ("m3",), ("m1",)),
        LearningCase("train_piano", "piano", _load("data/agent_conversation.json"), "c1", _recent("Falamos do Piano V e do jazz."), "Por que eu escolhi o Piano V?", ("m6",), ("m5",)),
        LearningCase("train_pack", "pack", _load("data/agent_conversation.json"), "c1", _recent("Falamos do pack."), "Como eu divulguei o pack?", ("m7",), ("m7",)),
        LearningCase("train_duas", "serum", _load("data/agent_conversation.json", "benchmark/extra_two.json"), "c1", _recent("Falamos do Serum 2 e do bass."), "Por que eu escolhi o preset Serum 2?", ("m2", "m10"), ("m1",)),
    ]
    evaluation = [
        LearningCase("eval_bicicleta", "bicicleta", _load("data/agent_conversation.json"), "c2", _recent("Falamos de bicicleta."), "Por que comprei a bicicleta?", ("p2",), ("p1",), split="eval"),
        LearningCase("eval_correcao", "correcao", _load("data/agent_conversation.json", "benchmark/v2_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m20",), ("m20",), split="eval"),
        LearningCase("eval_preco_mensagem", "preco", _load("data/agent_conversation.json", "benchmark/v2_price_message.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m20",), memory_limit=90, split="eval"),
        LearningCase("eval_preco_resposta", "preco", _load("data/agent_conversation.json", "benchmark/v2_price_reply.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m1",), memory_limit=110, split="eval"),
        LearningCase("eval_historico", "historico", _load("data/agent_conversation.json"), "c1", (RecentTurn("Porque eu já tinha comprado a licença.", id="m2"), RecentTurn("Falamos do Serum 2.")), "Por que eu escolhi o preset Serum 2?", ("m2",), (), split="eval"),
    ]
    return train, evaluation


def _novelty(message: Message, recent: Sequence[RecentTurn]) -> float:
    estimate = estimate_utility(
        message, 1.0, 1.0, [turn.text for turn in recent], ()
    )
    return estimate.novelty


def _features(
    message: Message,
    candidate,
    seed: Message,
    focus: str,
    recent: Sequence[RecentTurn],
    attribute: str | None,
    targets: frozenset[str],
    by_id: Mapping[str, Message],
) -> tuple[list[float], float]:
    evidence = float(candidate.evidence.get(focus) or 0.0)
    attribute_support = 0.0
    if attribute == ATTRIBUTE_PRICE:
        attribute_support = 1.0 if message_supports_attribute(
            message, attribute, targets, by_id
        ) else 0.0
    values = [
        evidence,
        float(candidate.association),
        1.0 if link_between(message, seed) else 0.0,
        _novelty(message, recent),
        1.0 - float(candidate.distance),
        attribute_support,
    ]
    return values, attribute_support


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def train_model(
    cases: Sequence[LearningCase],
    config: Mapping[str, Any],
    epochs: int = 600,
    learning_rate: float = 0.5,
    l2: float = 0.01,
    positive_weight: float = 3.0,
) -> dict[str, Any]:
    examples: list[tuple[list[float], int]] = []
    positives = 0
    for case in cases:
        recall_result = recall(
            case.messages,
            case.question,
            [turn.text for turn in case.recent],
            config,
            conversation_id=case.conversation_id,
        )
        seed = recall_result.seed
        attribute = requested_attribute(case.question)
        targets = attribute_targets(case.question)
        by_id = {message.id: message for message in case.messages}
        for candidate in recall_result.candidates:
            values, _ = _features(
                candidate.message,
                candidate,
                seed,
                recall_result.focus,
                case.recent,
                attribute,
                targets,
                by_id,
            )
            label = 1 if candidate.message.id in case.expected_ids else 0
            positives += label
            examples.append((values, label))

    weights = [0.0] * len(FEATURES)
    bias = 0.0
    for _ in range(epochs):
        gradient = [0.0] * len(FEATURES)
        bias_gradient = 0.0
        for values, label in examples:
            score = sum(weight * value for weight, value in zip(weights, values)) + bias
            prediction = _sigmoid(score)
            error = (prediction - label) * (positive_weight if label else 1.0)
            for index, value in enumerate(values):
                gradient[index] += error * value
            bias_gradient += error
        for index in range(len(weights)):
            weights[index] -= learning_rate * (
                gradient[index] / len(examples) + l2 * weights[index]
            )
        bias -= learning_rate * bias_gradient / len(examples)
    return {
        "weights": dict(zip(FEATURES, [round(weight, 6) for weight in weights])),
        "bias": round(bias, 6),
        "examples": len(examples),
        "positives": positives,
    }


def _id_in_context(message_id: str, messages: Sequence[Mapping[str, str]]) -> bool:
    context = "\n".join(str(message["content"]) for message in messages[:-1])
    if f"id={message_id}" in context or f"[{message_id}]" in context:
        return True
    return False


def evaluate_case(
    case: LearningCase,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    learned: bool,
) -> dict[str, Any]:
    recall_result = recall(
        case.messages,
        case.question,
        [turn.text for turn in case.recent],
        config,
        conversation_id=case.conversation_id,
    )
    seed = recall_result.seed
    attribute = requested_attribute(case.question)
    targets = attribute_targets(case.question)
    support = (
        attribute_supporting_ids(
            case.messages,
            attribute,
            case.conversation_id,
            targets,
            {message.id: message for message in case.messages},
        )
        if attribute
        else frozenset()
    )
    by_id = {message.id: message for message in case.messages}
    weights = model["weights"]
    candidates = []
    for candidate in recall_result.candidates:
        message = candidate.message
        if attribute and message.id not in support:
            continue
        values, attribute_support = _features(
            message,
            candidate,
            seed,
            recall_result.focus,
            case.recent,
            attribute,
            targets,
            by_id,
        )
        if learned:
            score = _sigmoid(
                sum(
                    weights[name] * value
                    for name, value in zip(FEATURES, values)
                )
                + model["bias"]
            )
        else:
            evidence = values[0]
            score = candidate.association * evidence
            if attribute_support:
                score = max(score, candidate.association)
        evidence = {**candidate.evidence, recall_result.focus: 1.0}
        candidates.append(
            replace(
                candidate,
                evidence=evidence,
                attention=score,
                within_radius=True,
            )
        )
    recent = case.recent
    base = build_base_messages(
        str(config.get("instructions", "")), recent, case.question
    )
    counter = EstimatedChatCounter(SimulatedCounter())
    base_tokens = counter.count_messages(base)
    window = base_tokens + 4096 + case.memory_limit
    result = select_chat_context(
        instructions=str(config.get("instructions", "")),
        recent_turns=recent,
        current_turn=case.question,
        seed=seed,
        focus=recall_result.focus,
        candidates=candidates,
        window=window,
        output_reserve=4096,
        counter=counter,
        top_k=int(config.get("top_k", 3)),
        threshold=float(config.get("attention_threshold", 0.05)),
        include_seed=True,
        use_utility=False,
    )
    selected = list(result.diagnostics.selected_ids)
    if result.diagnostics.seed_included:
        selected.append(seed.id)
    full = all(_id_in_context(message_id, result.messages) for message_id in case.expected_ids)
    irrelevant = [
        message_id
        for message_id in selected
        if message_id not in case.expected_ids
        and message_id not in case.dependency_ids
    ]
    candidate_ids = {candidate.message.id for candidate in recall_result.candidates}
    candidate_ids.add(seed.id)
    missing_candidates = [
        message_id for message_id in case.expected_ids if message_id not in candidate_ids
    ]
    return {
        "case": case.id,
        "family": case.family,
        "full": full,
        "selected": selected,
        "irrelevant": irrelevant,
        "additional_tokens": result.diagnostics.memory_tokens,
        "tokens": result.diagnostics.used_tokens,
        "missing_candidates": missing_candidates,
        "weights_used": "learned" if learned else "fixed",
    }


def run_experiment(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    train_cases, eval_cases = cases()
    model = train_model(train_cases, config)
    evaluation = []
    for case in eval_cases:
        fixed = evaluate_case(case, model, config, learned=False)
        learned = evaluate_case(case, model, config, learned=True)
        evaluation.append(
            {
                "case": case.id,
                "family": case.family,
                "fixed": fixed,
                "learned": learned,
                "changed": fixed["selected"] != learned["selected"]
                or fixed["full"] != learned["full"],
                "improved": learned["full"] and not fixed["full"],
                "regressed": fixed["full"] and not learned["full"],
            }
        )
    def aggregate(side: str) -> dict[str, Any]:
        items = [entry[side] for entry in evaluation]
        return {
            "cases": len(items),
            "full": sum(item["full"] for item in items),
            "irrelevant": sum(len(item["irrelevant"]) for item in items),
            "avg_additional_tokens": round(
                sum(item["additional_tokens"] for item in items) / len(items), 2
            ),
            "missing_candidates": sum(
                len(item["missing_candidates"]) for item in items
            ),
        }
    return {
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "train_families": sorted({case.family for case in train_cases}),
        "eval_families": sorted({case.family for case in eval_cases}),
        "model": model,
        "fixed": aggregate("fixed"),
        "learned": aggregate("learned"),
        "evaluation": evaluation,
        "improved": [entry["case"] for entry in evaluation if entry["improved"]],
        "regressed": [entry["case"] for entry in evaluation if entry["regressed"]],
        "changed": [entry["case"] for entry in evaluation if entry["changed"]],
    }
