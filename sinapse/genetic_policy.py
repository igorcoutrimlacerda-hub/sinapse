"""Otimização genética de parâmetros da política de memória.

Espaço de parâmetros:
- disparo da curiosidade: diferença mínima entre a melhor ação extra e stop [0, 0.5];
- ganho mínimo para manter uma memória na redução [0, 0.15];
- peso da redundância no ganho [0.2, 1.0].

Sensibilidade e orçamento são regras fixas. A referência é mais_treino com o
seletor de contexto reduzido.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .config import load_config
from .budget import RecentTurn
from .context_reduction import (
    _extra_eval_cases,
    _initial_state,
    _reduction,
    _run_action,
    _train_more_model,
)
from .curiosity_learning import (
    ACTIONS,
    CuriosityCase,
    _features,
    _id_in_messages,
    _load,
    _recent,
)
from .curiosity_round2 import probabilities
from .curiosity_round3 import _eval_cases as calibration_cases
from .memory import requires_seed
from .paired_answer import _cases as known_cases

BASE = "data/agent_conversation.json"


@dataclass(frozen=True)
class Policy:
    trigger: float
    min_gain: float
    redundancy: float

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE = Policy(0.0, 0.0, 0.6)
BOUNDS = {
    "trigger": (0.0, 0.5),
    "min_gain": (0.0, 0.15),
    "redundancy": (0.2, 1.0),
}


def _unseen_cases() -> list[CuriosityCase]:
    return [
        CuriosityCase(
            "new_pack_quantity",
            "pack_quantidade",
            _load(BASE),
            "c1",
            _recent("Quero saber quantos presets tem o pack."),
            "E aquilo do pack?",
            ("m8",),
            ("m7",),
            "m7",
            ("m8",),
            split="unseen",
            memory_limit=120,
        ),
        CuriosityCase(
            "new_pack_ambiguous",
            "pack_ambiguo",
            _load(BASE),
            "c1",
            _recent("Falamos do pack."),
            "E aquilo do pack?",
            ("m8", "m9"),
            ("m7",),
            "m7",
            ("m8",),
            split="unseen",
            memory_limit=160,
        ),
        CuriosityCase(
            "new_self_price",
            "autossuficiente",
            _load(BASE, "benchmark/v2_price_message.json"),
            "c1",
            _recent("Falamos do Serum 2."),
            "Quanto custou o Serum 2?",
            ("m20",),
            ("m20",),
            "m20",
            (),
            split="unseen",
            memory_limit=90,
        ),
        CuriosityCase(
            "new_history_bike",
            "historico",
            _load(BASE),
            "c2",
            (
                RecentTurn("Porque a trilha é mais curta.", id="p2"),
                RecentTurn("Falamos de bicicleta."),
            ),
            "Por que comprei a bicicleta?",
            ("p2",),
            (),
            "p1",
            (),
            split="unseen",
        ),
        CuriosityCase(
            "new_restricted",
            "sensibilidade",
            _load(BASE),
            "c1",
            _recent("Falamos do Serum 2."),
            "Por que eu escolhi o preset Serum 2?",
            (),
            (),
            "m1",
            ("m2",),
            split="unseen",
            preferences=("Não quero falar sobre a licença do Serum 2.",),
        ),
    ]


def _clip(policy: Policy) -> Policy:
    return Policy(
        trigger=min(BOUNDS["trigger"][1], max(BOUNDS["trigger"][0], policy.trigger)),
        min_gain=min(BOUNDS["min_gain"][1], max(BOUNDS["min_gain"][0], policy.min_gain)),
        redundancy=min(
            BOUNDS["redundancy"][1], max(BOUNDS["redundancy"][0], policy.redundancy)
        ),
    )


def _sample(rng: random.Random) -> Policy:
    return Policy(
        trigger=rng.uniform(*BOUNDS["trigger"]),
        min_gain=rng.uniform(*BOUNDS["min_gain"]),
        redundancy=rng.uniform(*BOUNDS["redundancy"]),
    )


def _mutate(rng: random.Random, policy: Policy) -> Policy:
    return _clip(
        Policy(
            trigger=policy.trigger + rng.gauss(0, 0.08),
            min_gain=policy.min_gain + rng.gauss(0, 0.03),
            redundancy=policy.redundancy + rng.gauss(0, 0.15),
        )
    )


def _choose_action(model, state, policy: Policy) -> str:
    scores = probabilities(model, _features(state))
    stop = scores["stop"]
    extras = {action: scores[action] for action in ("follow_links", "other_theme", "correction")}
    action, score = max(extras.items(), key=lambda item: item[1])
    if score >= stop + policy.trigger:
        return action
    return "stop"


def evaluate_policy(
    policy: Policy,
    cases: Sequence[CuriosityCase],
    model,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    full_cases = 0
    dependencies_lost = 0
    violations = 0
    functionless_total = 0
    tokens_total = 0
    candidates_total = 0
    extra_searches = 0
    details = []
    for case in cases:
        state = _initial_state(case, config)
        action = _choose_action(model, state, policy)
        current = _run_action(action, state, config)
        reduced = _reduction(
            state,
            current,
            config,
            min_gain=policy.min_gain,
            min_novelty=0.55,
            redundancy_weight=policy.redundancy,
        )
        selected = reduced["selected"]
        expected_present = [
            message_id
            for message_id in case.expected_ids
            if _id_in_messages(message_id, reduced["messages"])
        ]
        by_id = {message.id: message for message in case.messages}
        necessary_dependencies = []
        for dependency in case.dependency_ids:
            dependency_message = by_id.get(dependency)
            if dependency_message is None:
                continue
            for expected in case.expected_ids:
                evidence = by_id.get(expected)
                if evidence is not None and requires_seed(evidence, dependency_message):
                    necessary_dependencies.append(dependency)
                    break
        deps_present = [
            message_id
            for message_id in necessary_dependencies
            if _id_in_messages(message_id, reduced["messages"])
        ]
        expected_ok = len(expected_present) == len(case.expected_ids)
        deps_ok = len(deps_present) == len(necessary_dependencies)
        full = expected_ok and deps_ok
        if full:
            full_cases += 1
        dependencies_lost += len(necessary_dependencies) - len(deps_present)
        violation_ids = sorted(set(selected) & set(state.blocked))
        violations += len(violation_ids)
        functionless = [
            message_id
            for message_id in selected
            if message_id not in case.expected_ids
            and message_id not in case.dependency_ids
        ]
        functionless_total += len(functionless)
        tokens_total += reduced["memory_tokens"]
        candidates_total += current["extras"]
        if action != "stop":
            extra_searches += 1
        details.append(
            {
                "case": case.id,
                "family": case.family,
                "action": action,
                "full": full,
                "expected": expected_present,
                "dependencies": deps_present,
                "functionless": functionless,
                "tokens": reduced["memory_tokens"],
                "candidates": current["extras"],
                "violations": violation_ids,
                "selected": selected,
            }
        )
    key = (
        violations,
        -full_cases,
        functionless_total,
        tokens_total,
        candidates_total,
    )
    return {
        "policy": policy.as_dict(),
        "key": list(key),
        "full_cases": full_cases,
        "cases": len(cases),
        "dependencies_lost": dependencies_lost,
        "violations": violations,
        "functionless": functionless_total,
        "tokens": tokens_total,
        "candidates": candidates_total,
        "extra_searches": extra_searches,
        "details": details,
    }


def random_search(
    evaluations: int, seed: int, cases, model, config
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    best = None
    history = []
    for index in range(evaluations):
        policy = _sample(rng)
        result = evaluate_policy(policy, cases, model, config)
        result["evaluation"] = index + 1
        history.append({"evaluation": index + 1, "policy": result["policy"], "key": result["key"]})
        if best is None or tuple(result["key"]) < tuple(best["key"]):
            best = result
    assert best is not None
    return best, history


def genetic_search(
    evaluations: int,
    seed: int,
    cases,
    model,
    config,
    population_size: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    population = [_sample(rng) for _ in range(population_size)]
    scored = [(evaluate_policy(policy, cases, model, config), policy) for policy in population]
    evaluated = population_size
    best = min((result for result, _ in scored), key=lambda result: tuple(result["key"]))
    history = [
        {"evaluation": index + 1, "policy": result["policy"], "key": result["key"]}
        for index, (result, _) in enumerate(scored)
    ]
    while evaluated < evaluations:
        scored.sort(key=lambda item: tuple(item[0]["key"]))
        parents = [policy for _, policy in scored[:2]]
        children = []
        while len(children) < population_size and evaluated + len(children) < evaluations:
            child = _mutate(rng, parents[len(children) % 2])
            children.append(child)
        for child in children:
            result = evaluate_policy(child, cases, model, config)
            evaluated += 1
            history.append(
                {"evaluation": evaluated, "policy": result["policy"], "key": result["key"]}
            )
            if tuple(result["key"]) < tuple(best["key"]):
                best = result
            scored.append((result, child))
        scored.sort(key=lambda item: tuple(item[0]["key"]))
        scored = scored[:population_size]
    return best, history


def run_genetic_experiment(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config())
    model = _train_more_model(config)
    calibration = list(calibration_cases())
    known = list(known_cases())
    unseen = _unseen_cases()
    seeds = (11, 22, 33)
    evaluations = 24

    reference_eval = evaluate_policy(REFERENCE, calibration, model, config)
    random_runs = []
    genetic_runs = []
    for seed in seeds:
        random_best, random_history = random_search(
            evaluations, seed, calibration, model, config
        )
        genetic_best, genetic_history = genetic_search(
            evaluations, seed, calibration, model, config
        )
        random_runs.append({"seed": seed, "best": random_best, "history": random_history})
        genetic_runs.append({"seed": seed, "best": genetic_best, "history": genetic_history})
    best_random = min(
        (run["best"] for run in random_runs), key=lambda result: tuple(result["key"])
    )
    best_genetic = min(
        (run["best"] for run in genetic_runs), key=lambda result: tuple(result["key"])
    )
    frozen = {
        "reference": REFERENCE,
        "random": Policy(**best_random["policy"]),
        "genetic": Policy(**best_genetic["policy"]),
    }
    evaluations_out = {}
    for name, policy in frozen.items():
        evaluations_out[name] = {
            "known": evaluate_policy(policy, known, model, config),
            "unseen": evaluate_policy(policy, unseen, model, config),
        }
    return {
        "seeds": seeds,
        "evaluations_per_seed": evaluations,
        "calibration_cases": len(calibration),
        "known_cases": len(known),
        "unseen_cases": len(unseen),
        "bounds": BOUNDS,
        "reference": reference_eval,
        "random_runs": random_runs,
        "genetic_runs": genetic_runs,
        "best_random": best_random,
        "best_genetic": best_genetic,
        "frozen_policies": {name: policy.as_dict() for name, policy in frozen.items()},
        "evaluations": evaluations_out,
    }
