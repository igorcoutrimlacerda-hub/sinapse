"""Segunda rodada da curiosidade aprendida.

Compara regra fixa, modelo anterior, mais treino com sinais antigos e o mesmo
treino com força de vínculo (respostas diretas elegíveis à semente).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .budget import RecentTurn
from .config import load_config
from .text import normalize_text

ROOT = Path(__file__).resolve().parents[1]
BASE = "data/agent_conversation.json"


def _extra_training() -> list[CuriosityCase]:
    return [
        CuriosityCase("tr_short_motive", "serum", _load(BASE), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o Serum 2?", ("m2",), ("m1",), "m1", ("m2",)),
        CuriosityCase("tr_link_distractor", "procedimento", _load(BASE), "c1", _recent("Falamos do Serum 2 e do bass."), "Como eu testei o bass?", ("m3",), ("m1",), "m1", ("m3",)),
        CuriosityCase("tr_history", "historico", _load(BASE), "c1", (RecentTurn("Porque eu já tinha comprado a licença.", id="m2"),), "Por que eu escolhi o Serum 2?", ("m2",), (), "m1", ()),
        CuriosityCase("tr_restricted", "sensibilidade", _load(BASE), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), (), "m1", ("m2",), preferences=("Não quero falar sobre a licença do Serum 2.",)),
    ]


def _eval_cases() -> list[CuriosityCase]:
    return [
        CuriosityCase("ev_follow", "bicicleta", _load(BASE), "c2", _recent("Falamos de bicicleta."), "Por que comprei a bicicleta?", ("p2",), ("p1",), "p1", ("p2",), split="eval"),
        CuriosityCase("ev_follow_piano", "piano", _load(BASE), "c1", _recent("Falamos do Serum 2 e do Piano V."), "E aquilo do piano?", ("m6",), ("m5",), "m5", ("m6",), split="eval"),
        CuriosityCase("ev_link_distractor", "procedimento", _load(BASE), "c1", _recent("Falamos do Serum 2 e do bass."), "Como eu testei o bass?", ("m3",), ("m1",), "m1", ("m3",), split="eval"),
        CuriosityCase("ev_history", "historico", _load(BASE), "c1", (RecentTurn("Porque eu já tinha comprado a licença.", id="m2"), RecentTurn("Falamos do Serum 2.")), "Por que eu escolhi o Serum 2?", ("m2",), (), "m1", (), split="eval"),
        CuriosityCase("ev_correction", "correcao", _load(BASE, "benchmark/extra_correction.json"), "c1", _recent("Escolhi o Serum 2 para o bass."), "Qual preset eu uso agora?", ("m10",), ("m1",), "m1", ("m10",), split="eval"),
        CuriosityCase("ev_restricted", "sensibilidade", _load(BASE), "c1", _recent("Falamos do Serum 2."), "Por que eu escolhi o preset Serum 2?", (), (), "m1", ("m2",), split="eval", preferences=("Não quero falar sobre a licença do Serum 2.",)),
        CuriosityCase("ev_price", "preco", _load(BASE, "benchmark/v2_price_reply.json"), "c1", _recent("Falamos do Serum 2."), "Quanto custou o Serum 2?", ("m20",), ("m1",), "m1", ("m20",), split="eval", memory_limit=110),
    ]


def link_signal(state) -> tuple[float, float]:
    started = time.perf_counter()
    count = sum(
        1
        for message in state.case.messages
        if message.reply_to == state.seed.id and message.id not in state.blocked
    )
    elapsed = (time.perf_counter() - started) * 1000
    return min(1.0, count / 3.0), elapsed


def features_with_link(state) -> tuple[list[float], float]:
    signal, cost = link_signal(state)
    return _features(state) + [signal], cost


def probabilities(model, features: Sequence[float]) -> dict[str, float]:
    import math

    scores = [
        sum(weight * value for weight, value in zip(model["weights"][index], features))
        + model["biases"][index]
        for index in range(len(ACTIONS))
    ]
    peak = max(scores)
    exp = [math.exp(score - peak) for score in scores]
    total = sum(exp)
    return {
        action: round(exp[index] / total, 4)
        for index, action in enumerate(ACTIONS)
    }


def _expected_and_context(case: CuriosityCase, result: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    from .curiosity_learning import _id_in_messages

    expected = [
        message_id
        for message_id in case.expected_ids
        if _id_in_messages(message_id, result["messages"])
    ]
    context = [
        message_id
        for message_id in case.dependency_ids
        if _id_in_messages(message_id, result["messages"])
    ]
    selected = result["selected"]
    functionless = [
        message_id
        for message_id in selected
        if message_id not in expected and message_id not in context
    ]
    return expected, context, functionless


def _aggregate(entries: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    items = [entry[key] for entry in entries]
    extra_searches = sum(1 for item in items if item["action"] != "stop")
    no_gain = sum(
        1
        for entry in entries
        if entry[key]["action"] != "stop"
        and len(entry[key]["expected"]) <= len(entry["stop"]["expected"])
    )
    triggered = sum(1 for item in items if item["action"] != "stop")
    latencies = sorted(item["elapsed_ms"] for item in items)
    return {
        "expected": sum(len(item["expected"]) for item in items),
        "context": sum(len(item["context"]) for item in items),
        "functionless": sum(len(item["functionless"]) for item in items),
        "extra_searches": extra_searches,
        "no_gain_searches": no_gain,
        "queries_triggered": triggered,
        "cases": len(items),
        "candidates": sum(item["extras"] for item in items),
        "memory_tokens": sum(item["memory_tokens"] for item in items),
        "tokens": sum(item["tokens"] for item in items),
        "latency_median_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else 0.0,
    }


def run_round2(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    base_train, _ = base_cases()
    train_cases = list(base_train) + _extra_training()
    eval_cases = _eval_cases()

    def rows(use_link: bool, cases: Sequence[CuriosityCase]):
        result = []
        for case in cases:
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
            if use_link:
                feature_values, _ = features_with_link(state)
            else:
                feature_values = _features(state)
            result.append((feature_values, ACTIONS.index(best)))
        return result

    old_model = train_policy(rows(False, base_train))
    more_model = train_policy(rows(False, train_cases))
    link_model = train_policy(rows(True, train_cases))

    entries = []
    signal_cost = 0.0
    for case in eval_cases:
        state = _initial_state(case, config)
        link_features, cost = features_with_link(state)
        signal_cost += cost
        old_features = _features(state)
        stop = _run_action("stop", state, config)
        fixed = _run_action(fixed_action(state), state, config)
        old_action = max(ACTIONS, key=lambda action: probabilities(old_model, old_features)[action])
        more_action = max(ACTIONS, key=lambda action: probabilities(more_model, old_features)[action])
        link_action = max(ACTIONS, key=lambda action: probabilities(link_model, link_features)[action])
        runs = {
            "off": stop,
            "fixed": fixed,
            "old": _run_action(old_action, state, config),
            "more": _run_action(more_action, state, config),
            "link": _run_action(link_action, state, config),
        }
        item: dict[str, Any] = {
            "case": case.id,
            "family": case.family,
            "features_old": dict(
                zip(
                    ("ambiguidade", "faltante", "conflito", "vinculos", "candidatos", "atributo_disponivel", "atributo_pedido", "restricao"),
                    old_features,
                )
            ),
            "features_link": {
                "respostas_diretas": link_features[-1],
                "custo_sinal_ms": round(cost, 3),
            },
            "scores_old": probabilities(old_model, old_features),
            "scores_link": probabilities(link_model, link_features),
            "eligible_actions": _available_actions(state),
            "all_actions": {},
            "actions": {},
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
            item["actions"][run["action"]] = {
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
        "old": _aggregate(entries, "old"),
        "more": _aggregate(entries, "more"),
        "link": _aggregate(entries, "link"),
    }
    families: dict[str, Any] = {}
    for family in sorted({entry["family"] for entry in entries}):
        subset = [entry for entry in entries if entry["family"] == family]
        families[family] = {
            condition: sum(len(entry[condition]["expected"]) for entry in subset)
            for condition in ("off", "fixed", "old", "more", "link")
        }
    return {
        "train_base": len(base_train),
        "train_extended": len(train_cases),
        "eval_cases": len(eval_cases),
        "signal_avg_cost_ms": round(signal_cost / len(eval_cases), 3),
        "summary": summary,
        "families": families,
        "entries": entries,
        "diag_follow": next(entry for entry in entries if entry["case"] == "ev_follow"),
    }
