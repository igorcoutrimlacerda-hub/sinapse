"""Seleção por tema e propósito: referência vs política de utilidade aprendida.

Cada turno declara propósitos (quantidade, divulgação, motivo, correção), cada
um com tema, evidência necessária e contexto. A política aprende, por tipo de
propósito, uma utilidade a partir de sinais disponíveis em execução. IDs
esperados são usados apenas como rótulos de treino e avaliação.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import (
    RecentTurn,
    SimulatedCounter,
    render_evidence_block,
    render_seed_block,
    wrap_memory,
)
from .chat import EstimatedChatCounter, with_memory
from .config import load_config
from .context_reduction import _initial_state, _reduction, _run_action, _train_more_model
from .curiosity_learning import ACTIONS, CuriosityCase, _id_in_messages, _load, _recent
from .curiosity_round2 import probabilities
from .genetic_policy import _choose_action
from .memory import link_between, message_tokens, requires_seed
from .models import Candidate, Message
from .text import normalize_text, tokenize
from .utility import estimate_utility

BASE = "data/agent_conversation.json"
KINDS = ("quantidade", "divulgacao", "motivo", "correcao", "preco")
KIND_TERMS = {
    "quantidade": ("quantos", "quantas", "quantidade", "presets", "oito", "doze", "8", "12"),
    "divulgacao": ("divulgar", "divulguei", "divulgacao", "forum", "preparei", "pack"),
    "motivo": ("porque", "motivo", "licenca", "promocao", "indicacao", "amigo", "escolhi"),
    "correcao": ("troquei", "deixei", "agora", "pigments", "corrigi", "mudei"),
    "preco": ("custou", "custa", "preco", "valor", "paguei", "reais", "1200"),
}


@dataclass(frozen=True)
class Purpose:
    id: str
    kind: str
    theme: str
    dimension: str
    evidence_ids: tuple[str, ...]
    context_ids: tuple[str, ...] = ()
    answer_terms: tuple[str, ...] = ()
    unavailable: bool = False


@dataclass(frozen=True)
class PurposeCase:
    id: str
    family: str
    messages: tuple[Message, ...]
    conversation_id: str
    recent: tuple[RecentTurn, ...]
    question: str
    purposes: tuple[Purpose, ...]
    seed_id: str
    memory_limit: int = 140
    preferences: tuple[str, ...] = ()
    split: str = "train"
    hidden_ids: tuple[str, ...] = ()
    priority_purpose_id: str | None = None

    @property
    def expected_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                message_id
                for purpose in self.purposes
                for message_id in purpose.evidence_ids
            )
        )

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                message_id
                for purpose in self.purposes
                for message_id in purpose.context_ids
            )
        )


def _cases() -> tuple[list[PurposeCase], list[PurposeCase]]:
    base = _load(BASE)
    train = [
        PurposeCase("tr_pack_quantity", "pack", base, "c1", _recent("Falamos do pack."), "Quantos presets tem o pack?", (Purpose("q", "quantidade", "pack", "o_que", ("m8",), ("m7",)),), "m7"),
        PurposeCase("tr_pack_quantity_unlinked", "pack", base, "c1", _recent("Falamos do pack e do Serum 2."), "Quantos presets tem o pack?", (Purpose("q", "quantidade", "pack", "o_que", ("m8",), ()),), "m1"),
        PurposeCase("tr_pack_disclosure", "pack", base, "c1", _recent("Falamos do pack."), "Como eu divulguei o pack?", (Purpose("d", "divulgacao", "pack", "como", ("m7",)),), "m7"),
        PurposeCase("tr_pack_both", "pack", base, "c1", _recent("Falamos do pack."), "Quantos presets tem o pack e como eu divulguei?", (Purpose("q", "quantidade", "pack", "o_que", ("m8",), ("m7",)), Purpose("d", "divulgacao", "pack", "como", ("m7",))), "m7", memory_limit=180),
        PurposeCase("tr_serum_motive", "serum", base, "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o Serum 2?", (Purpose("m", "motivo", "serum", "por_que", ("m2",), ("m1",)),), "m1"),
        PurposeCase("tr_two_themes", "misto", base, "c1", _recent("Falamos do Serum 2 e do pack."), "Por que escolhi o Serum 2 e quantos presets tem o pack?", (Purpose("m", "motivo", "serum", "por_que", ("m2",), ("m1",)), Purpose("q", "quantidade", "pack", "o_que", ("m8",))), "m1", memory_limit=180),
        PurposeCase("tr_correction", "correcao", _load(BASE, "benchmark/extra_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", (Purpose("c", "correcao", "pigments", "o_que", ("m10",)),), "m1"),
        PurposeCase("tr_price_bike", "bicicleta", _load(BASE, "benchmark/price_bicycle.json"), "c2", _recent("Falamos de bicicleta."), "Quanto custou a bicicleta?", (Purpose("p", "preco", "bicicleta", "o_que", ("p3",)),), "p1"),
    ]
    evaluation = [
        PurposeCase("ev_bike_motive", "bicicleta", base, "c2", _recent("Falamos de bicicleta."), "Por que comprei a bicicleta?", (Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), ("p1",)),), "p1", split="eval"),
        PurposeCase("ev_piano_motive", "piano", base, "c1", _recent("Falamos do Piano V."), "Por que escolhi o Piano V?", (Purpose("m", "motivo", "piano", "por_que", ("m6",), ("m5",)),), "m5", split="eval"),
        PurposeCase("ev_pack_both", "pack", base, "c1", _recent("Falamos do pack."), "Me diga a quantidade do pack e como foi a divulgação.", (Purpose("q", "quantidade", "pack", "o_que", ("m8",), ("m7",)), Purpose("d", "divulgacao", "pack", "como", ("m7",))), "m7", memory_limit=180, split="eval"),
        PurposeCase("ev_two_themes", "misto", base, "c1", _recent("Falamos do Piano V e do pack."), "Por que escolhi o Piano V e quantos presets tem o pack?", (Purpose("m", "motivo", "piano", "por_que", ("m6",), ("m5",)), Purpose("q", "quantidade", "pack", "o_que", ("m8",))), "m5", memory_limit=200, split="eval"),
        PurposeCase("ev_serum_contrast", "serum", base, "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o Serum 2?", (Purpose("m", "motivo", "serum", "por_que", ("m2",), ("m1",)),), "m1", split="eval"),
        PurposeCase("ev_history_bike", "historico", base, "c2", (RecentTurn("Porque a trilha é mais curta.", id="p2"), RecentTurn("Falamos de bicicleta.")), "Por que comprei a bicicleta?", (Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), ()),), "p1", split="eval"),
        PurposeCase("ev_restricted", "sensibilidade", base, "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), "m1", preferences=("Não quero falar sobre a licença do Serum 2.",), split="eval"),
    ]
    return train, evaluation


def _candidate_message(message_id: str, messages: Sequence[Message]) -> Message | None:
    return next((message for message in messages if message.id == message_id), None)


def _purpose_compatible(message: Message, purpose: Purpose) -> bool:
    text = normalize_text(message.text)
    if purpose.kind == "quantidade":
        return bool(message.values("o_que")) or any(
            term in text for term in ("presets", "oito", "doze", "quantidade")
        ) or any(character.isdigit() for character in message.text)
    if purpose.kind == "divulgacao":
        return any(message.values(dimension) for dimension in ("como", "onde", "para_que")) or any(
            term in text for term in ("divulgar", "divulguei", "forum", "preparei")
        )
    if purpose.kind == "motivo":
        return bool(message.values("por_que"))
    if purpose.kind == "correcao":
        return any(
            term in text for term in ("troquei", "deixei", "agora", "pigments", "corrigi", "mudei")
        )
    if purpose.kind == "preco":
        return any(message.values(dimension) for dimension in ("o_que", "por_que")) or any(
            term in text for term in ("custou", "preco", "valor", "paguei", "reais")
        )
    return True


def _features(
    message: Message,
    purpose: Purpose,
    question: str,
    seed: Message,
    recent: Sequence[RecentTurn],
    association: float = 0.9,
) -> list[float]:
    tokens = message_tokens(message)
    theme_tokens = tokenize(purpose.theme)
    query_tokens = tokenize(question)
    kind_terms = set(KIND_TERMS[purpose.kind])
    theme_match = len(tokens & theme_tokens) / len(theme_tokens) if theme_tokens else 0.0
    query_match = len(tokens & query_tokens) / len(tokens) if tokens else 0.0
    kind_match = len(tokens & kind_terms) / len(kind_terms) if kind_terms else 0.0
    dimension_match = 1.0 if message.values(purpose.dimension) else 0.0
    annotation = 1.0 if any(message.values(dimension) for dimension in message.coordinates) else 0.0
    linked = 1.0 if link_between(message, seed) else 0.0
    novelty = estimate_utility(message, 1.0, 1.0, [turn.text for turn in recent], ()).novelty
    number = 1.0 if any(character.isdigit() for character in message.text) or any(
        word in normalize_text(message.text) for word in ("oito", "dois", "doze", "doze")
    ) else 0.0
    return [
        theme_match,
        query_match,
        kind_match,
        dimension_match,
        annotation,
        linked,
        novelty,
        min(1.0, association),
        number,
    ]


def _train_kind(rows: Sequence[tuple[list[float], int]], epochs: int = 400, lr: float = 0.5, positive_weight: float = 2.5) -> dict[str, Any]:
    weights = [0.0] * len(rows[0][0])
    bias = 0.0
    for _ in range(epochs):
        grads = [0.0] * len(weights)
        bias_grad = 0.0
        for values, label in rows:
            score = sum(weight * value for weight, value in zip(weights, values)) + bias
            prediction = 1.0 / (1.0 + math.exp(-score))
            target = 1.0 if label == 2 else 0.0
            error = (prediction - target) * (positive_weight if target else 1.0)
            for index, value in enumerate(values):
                grads[index] += error * value
            bias_grad += error
        for index in range(len(weights)):
            weights[index] -= lr * grads[index] / len(rows)
        bias -= lr * bias_grad / len(rows)
    return {"weights": weights, "bias": bias}


def _score(model: Mapping[str, Any], values: Sequence[float]) -> float:
    raw = sum(weight * value for weight, value in zip(model["weights"], values)) + model["bias"]
    return 1.0 / (1.0 + math.exp(-raw))


def _train(train_cases: Sequence[PurposeCase], config: Mapping[str, Any]):
    rows: dict[str, list[tuple[list[float], int]]] = {kind: [] for kind in KINDS}
    for case in train_cases:
        state = _initial_state(case, config)
        by_id = {message.id: message for message in case.messages}
        candidates = list(state.candidates) + [
            Candidate(
                message=state.seed,
                distance=0.0,
                association=1.0,
                association_content=1.0,
                association_link=0.0,
                association_source="seed",
                evidence={dimension: None for dimension in ("o_que", "quem", "onde", "quando", "como", "por_que", "para_que")},
                attention=1.0,
                within_radius=True,
            )
        ]
        for purpose in case.purposes:
            for candidate in candidates:
                message = candidate.message
                if message.id in purpose.evidence_ids:
                    label = 2
                elif message.id in purpose.context_ids:
                    label = 1
                else:
                    label = 0
                rows[purpose.kind].append(
                    (
                        _features(message, purpose, case.question, state.seed, case.recent, candidate.association),
                        label,
                    )
                )
    return {kind: _train_kind(values) for kind, values in rows.items() if values}


def _select_purpose(
    case: PurposeCase,
    models: Mapping[str, Any],
    state,
    reference_candidates: Sequence[str],
) -> dict[str, Any]:
    by_id = {message.id: message for message in case.messages}
    ids = list(
        dict.fromkeys(
            [
                *[candidate.message.id for candidate in state.candidates],
                *reference_candidates,
                state.seed.id,
            ]
        )
    )
    candidates = [by_id[message_id] for message_id in ids if message_id in by_id]
    active_purposes = [
        purpose
        for purpose in case.purposes
        if not all(
            _id_in_messages(message_id, state.base_messages)
            for message_id in purpose.evidence_ids
        )
    ]
    if not active_purposes:
        return {
            "messages": state.base_messages,
            "selected": [],
            "memory_tokens": 0,
            "reasons": {},
        }
    theme_tokens: set[str] = set()
    for purpose in active_purposes:
        theme_tokens.update(tokenize(purpose.theme))
    selected: list[Message] = []
    covered: set[str] = set()
    reasons = {}
    scored: list[tuple[float, Message]] = []
    for message in candidates:
        if (
            message.id != state.seed.id
            and not link_between(message, state.seed)
            and not (message_tokens(message) & theme_tokens)
        ):
            continue
        scores = {}
        for purpose in active_purposes:
            model = models.get(purpose.kind)
            if model is None or purpose.id in covered:
                continue
            if not _purpose_compatible(message, purpose):
                continue
            scores[purpose.id] = _score(
                model,
                _features(message, purpose, case.question, state.seed, case.recent, 0.9),
            )
            if purpose.id == case.priority_purpose_id:
                scores[purpose.id] = min(1.0, scores[purpose.id] + 0.2)
        if not scores:
            continue
        best = max(scores.values())
        if best < 0.4:
            continue
        scored.append((best, message))
        reasons[message.id] = {purpose_id: round(score, 4) for purpose_id, score in scores.items()}
    scored.sort(key=lambda item: (-item[0], item[1].position))
    counter = EstimatedChatCounter(SimulatedCounter())
    for _, message in scored:
        scores = reasons[message.id]
        if not any(score >= 0.4 and purpose_id not in covered for purpose_id, score in scores.items()):
            continue
        trial = selected + [message] if message not in selected else list(selected)
        needs_seed = any(
            requires_seed(item, state.seed)
            for item in trial
            if item.id != state.seed.id
        )
        keep_seed = needs_seed or any(item.id == state.seed.id for item in trial)
        evidence = [item for item in trial if item.id != state.seed.id]
        blocks = []
        if keep_seed:
            blocks.append(render_seed_block(state.seed))
        for rank, item in enumerate(evidence, start=1):
            blocks.append(render_evidence_block(item, rank))
        messages = with_memory(state.base_messages, wrap_memory(blocks))
        used = counter.count_messages(messages)
        if used + state.reserve <= state.window:
            selected = trial
            for purpose_id, score in scores.items():
                if score >= 0.4:
                    covered.add(purpose_id)
    needs_seed = any(requires_seed(message, state.seed) for message in selected)
    keep_seed = needs_seed or any(message.id == state.seed.id for message in selected)
    evidence = [message for message in selected if message.id != state.seed.id]
    blocks = []
    if keep_seed:
        blocks.append(render_seed_block(state.seed))
    for rank, message in enumerate(evidence, start=1):
        blocks.append(render_evidence_block(message, rank))
    memory_text = wrap_memory(blocks)
    messages = with_memory(state.base_messages, memory_text)
    used = counter.count_messages(messages)
    return {
        "messages": messages,
        "selected": ([state.seed.id] if keep_seed else []) + [message.id for message in evidence],
        "memory_tokens": used - state.base_tokens,
        "reasons": reasons,
    }


def _evaluate(
    case: PurposeCase,
    state,
    selected_ids: Sequence[str],
    messages,
    memory_tokens: int,
) -> dict[str, Any]:
    coverage = {}
    for purpose in case.purposes:
        evidence_ok = all(_id_in_messages(message_id, messages) for message_id in purpose.evidence_ids)
        deps_ok = all(_id_in_messages(message_id, messages) for message_id in purpose.context_ids)
        coverage[purpose.id] = evidence_ok and deps_ok
    allowed = {
        message_id
        for purpose in case.purposes
        for message_id in purpose.evidence_ids + purpose.context_ids
    }
    distractions = [message_id for message_id in selected_ids if message_id not in allowed]
    deps_lost = [
        message_id
        for purpose in case.purposes
        for message_id in purpose.context_ids
        if not _id_in_messages(message_id, messages)
    ]
    return {
        "purposes": coverage,
        "all_purposes": all(coverage.values()) if coverage else True,
        "coverage_count": sum(coverage.values()),
        "purpose_count": len(coverage),
        "deps_lost": deps_lost,
        "distractions": distractions,
        "selected": list(selected_ids),
        "memory_tokens": memory_tokens,
    }


def run_purpose_benchmark(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    train_cases, eval_cases = _cases()
    models = _train(train_cases, config)
    model = _train_more_model(config)
    entries = []
    for case in eval_cases:
        state = _initial_state(case, config)
        policy = _choose_action(model, state, __import__("sinapse.genetic_policy", fromlist=["REFERENCE"]).REFERENCE)
        current = _run_action(policy, state, config)
        reduced = _reduction(state, current, config)
        reference_metrics = _evaluate(
            case,
            state,
            reduced["selected"],
            reduced["messages"],
            reduced["memory_tokens"],
        )
        purpose_selection = _select_purpose(
            case,
            models,
            state,
            current.get("candidate_ids", []),
        )
        purpose_metrics = _evaluate(
            case,
            state,
            purpose_selection["selected"],
            purpose_selection["messages"],
            purpose_selection["memory_tokens"],
        )
        entries.append(
            {
                "case": case.id,
                "family": case.family,
                "purposes": [purpose.id for purpose in case.purposes],
                "reference": reference_metrics,
                "purpose": purpose_metrics,
                "reasons": purpose_selection["reasons"],
            }
        )

    def aggregate(side: str) -> dict[str, Any]:
        items = [entry[side] for entry in entries]
        total_purposes = sum(item["purpose_count"] for item in items)
        covered = sum(item["coverage_count"] for item in items)
        return {
            "cases": len(items),
            "purpose_coverage": round(covered / total_purposes, 4) if total_purposes else 1.0,
            "all_purposes_cases": sum(item["all_purposes"] for item in items),
            "deps_lost": sum(len(item["deps_lost"]) for item in items),
            "distractions": sum(len(item["distractions"]) for item in items),
            "memory_tokens": sum(item["memory_tokens"] for item in items),
        }

    improvements = [
        entry["case"]
        for entry in entries
        if entry["purpose"]["all_purposes"] and not entry["reference"]["all_purposes"]
    ]
    regressions = [
        entry["case"]
        for entry in entries
        if entry["reference"]["all_purposes"] and not entry["purpose"]["all_purposes"]
    ]
    return {
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "reference": aggregate("reference"),
        "purpose": aggregate("purpose"),
        "entries": entries,
        "improvements": improvements,
        "regressions": regressions,
    }
