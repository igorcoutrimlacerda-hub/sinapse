"""Rodada 3: vínculo compatível com dimensão/atributo, antes da busca extra.

O indicador conta respostas diretas elegíveis à semente cuja anotação cobre a
dimensão pedida (ou o atributo pedido, no caso de preço). Ele é calculado antes
da busca extra, com sensibilidade aplicada. Um índice reverso por
``reply_to`` é construído e seu custo é medido.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attributes import (
    ATTRIBUTE_PRICE,
    attribute_targets,
    message_supports_attribute,
)
from .budget import RecentTurn
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
    fixed_action,
    train_policy,
)
from .curiosity_round2 import (
    _aggregate,
    _eval_cases as round2_eval_cases,
    _expected_and_context,
    _extra_training,
    probabilities,
)
from .config import load_config
from .models import Message

ROOT = Path(__file__).resolve().parents[1]
BASE = "data/agent_conversation.json"


def _training_cases() -> list[CuriosityCase]:
    base, _ = base_cases()
    return list(base) + _extra_training()


def _eval_cases() -> list[CuriosityCase]:
    return [
        CuriosityCase("ev_follow", "bicicleta", _load(BASE), "c2", _recent("Falamos de bicicleta."), "Por que comprei a bicicleta?", ("p2",), ("p1",), "p1", ("p2",), split="eval"),
        CuriosityCase("ev_follow_piano", "piano", _load(BASE), "c1", _recent("Falamos do Serum 2 e do Piano V."), "E aquilo do piano?", ("m6",), ("m5",), "m5", ("m6",), split="eval"),
        CuriosityCase("ev_correction", "correcao", _load(BASE, "benchmark/extra_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m10",), ("m1",), "m1", ("m10",), split="eval"),
        CuriosityCase("ev_price_link", "preco", _load(BASE, "benchmark/v2_price_reply.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m1",), "m1", ("m20",), split="eval", memory_limit=110),
        CuriosityCase("ev_many_links", "muitos_vinculos", _load(BASE, "benchmark/many_links.json", "benchmark/extra_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m10",), ("m1",), "m1", ("m10",), split="eval"),
        CuriosityCase("ev_other_theme", "pack", _load(BASE), "c1", _recent("Falamos do Serum 2 e do pack."), "E aquilo do pack?", ("m9",), ("m7",), "m7", ("m9",), split="eval", memory_limit=180),
        CuriosityCase("ev_history", "historico", _load(BASE), "c1", (RecentTurn("Porque eu já tinha comprado a licença.", id="m2"), RecentTurn("Falamos do Serum 2.")), "Por que eu escolhi o Serum 2?", ("m2",), (), "m1", (), split="eval"),
        CuriosityCase("ev_restricted", "sensibilidade", _load(BASE), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), (), "m1", ("m2",), split="eval", preferences=("Não quero falar sobre a licença do Serum 2.",)),
    ]


class ReplyIndex:
    def __init__(self, messages: Sequence[Message]) -> None:
        started = time.perf_counter()
        self.by_parent: dict[str, list[Message]] = {}
        for message in messages:
            if message.reply_to:
                self.by_parent.setdefault(message.reply_to, []).append(message)
        self.build_ms = (time.perf_counter() - started) * 1000
        self.entries = sum(len(items) for items in self.by_parent.values())


def compatible_signal(state) -> tuple[float, float, ReplyIndex]:
    started = time.perf_counter()
    index = ReplyIndex(state.case.messages)
    replies = index.by_parent.get(state.seed.id, [])
    by_id = {message.id: message for message in state.case.messages}
    targets = attribute_targets(state.case.question)
    count = 0
    for message in replies:
        if message.id in state.blocked:
            continue
        if state.attribute == ATTRIBUTE_PRICE:
            if message_supports_attribute(message, state.attribute, targets, by_id):
                count += 1
            continue
        dimensions = [
            dimension
            for dimension in message.coordinates
            if message.values(dimension)
        ]
        focus = state.recall_result.focus
        compatible = focus in dimensions or (
            focus == "o_que"
            and any(
                dimension in dimensions
                for dimension in ("o_que", "por_que", "como", "para_que")
            )
        )
        if compatible:
            count += 1
    query_ms = (time.perf_counter() - started) * 1000
    return min(1.0, count / 3.0), query_ms, index


def run_round3(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    train_cases = _training_cases()
    eval_cases = _eval_cases()

    def rows(use_compatible: bool):
        result = []
        for case in train_cases:
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
            features = _features(state)
            if use_compatible:
                signal, _, _ = compatible_signal(state)
                features = features + [signal]
            result.append((features, ACTIONS.index(best)))
        return result

    more_model = train_policy(rows(False))
    compatible_model = train_policy(rows(True))

    entries = []
    build_cost = 0.0
    query_cost = 0.0
    index_entries = 0
    for case in eval_cases:
        state = _initial_state(case, config)
        old_features = _features(state)
        signal, q_cost, index = compatible_signal(state)
        compatible_features = old_features + [signal]
        build_cost += index.build_ms
        query_cost += q_cost
        index_entries += index.entries
        runs = {
            "off": _run_action("stop", state, config),
            "fixed": _run_action(fixed_action(state), state, config),
        }
        more_action = max(
            ACTIONS, key=lambda action: probabilities(more_model, old_features)[action]
        )
        compatible_action = max(
            ACTIONS,
            key=lambda action: probabilities(compatible_model, compatible_features)[action],
        )
        runs["more"] = _run_action(more_action, state, config)
        runs["compatible"] = _run_action(compatible_action, state, config)
        item: dict[str, Any] = {
            "case": case.id,
            "family": case.family,
            "features_old": dict(
                zip(
                    ("ambiguidade", "faltante", "conflito", "vinculos", "candidatos", "atributo_disponivel", "atributo_pedido", "restricao"),
                    old_features,
                )
            ),
            "features_compatible": {
                "vinculo_compativel": signal,
                "indice_respostas": index.entries,
                "indice_build_ms": round(index.build_ms, 4),
                "consulta_ms": round(q_cost, 4),
            },
            "scores_more": probabilities(more_model, old_features),
            "scores_compatible": probabilities(compatible_model, compatible_features),
            "eligible_actions": _available_actions(state),
            "all_actions": {},
            "off": {},
            "stop": {},
        }
        for action in item["eligible_actions"]:
            run = _run_action(action, state, config)
            expected, context, functionless = _expected_and_context(case, run)
            item["all_actions"][action] = {
                "expected": expected,
                "context": context,
                "functionless": functionless,
                "extras": run["extras"],
                "memory_tokens": run["memory_tokens"],
                "tokens": run["tokens"],
                "elapsed_ms": run["elapsed_ms"],
            }
        for key, run in runs.items():
            expected, context, functionless = _expected_and_context(case, run)
            item[key] = {
                "action": run["action"],
                "expected": expected,
                "context": context,
                "functionless": functionless,
                "extras": run["extras"],
                "memory_tokens": run["memory_tokens"],
                "tokens": run["tokens"],
                "elapsed_ms": run["elapsed_ms"],
            }
        item["stop"] = item["off"]
        entries.append(item)

    summary = {
        "off": _aggregate(entries, "off"),
        "fixed": _aggregate(entries, "fixed"),
        "more": _aggregate(entries, "more"),
        "compatible": _aggregate(entries, "compatible"),
    }
    families: dict[str, Any] = {}
    for family in sorted({entry["family"] for entry in entries}):
        subset = [entry for entry in entries if entry["family"] == family]
        families[family] = {
            condition: sum(len(entry[condition]["expected"]) for entry in subset)
            for condition in ("off", "fixed", "more", "compatible")
        }
    return {
        "train_cases": len(train_cases),
        "eval_cases": len(eval_cases),
        "index_avg_build_ms": round(build_cost / len(eval_cases), 4),
        "index_avg_query_ms": round(query_cost / len(eval_cases), 4),
        "index_avg_entries": round(index_entries / len(eval_cases), 1),
        "summary": summary,
        "families": families,
        "entries": entries,
        "diag_piano": next(entry for entry in entries if entry["case"] == "ev_follow_piano"),
    }
