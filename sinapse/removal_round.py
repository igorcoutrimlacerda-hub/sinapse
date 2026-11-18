"""Executa a rodada de remoções dirigidas do plano v2.

Gera todas as requisições localmente, remove exatamente o item planejado do
contexto da política por propósito, reutiliza requisições idênticas e só então
faz as chamadas reais necessárias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter, render_evidence_block, render_seed_block, wrap_memory
from .chat import EstimatedChatCounter, with_memory
from .config import load_config
from .context_reduction import _initial_state, _reduction, _run_action, _train_more_model
from .curiosity_learning import _id_in_messages
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .genetic_policy import REFERENCE, _choose_action
from .purpose_answer import SimulatedPurposeClient, _checks, _render_selection
from .purpose_selection import Purpose, PurposeCase, _cases as purpose_training_cases, _select_purpose, _train, _load, _recent

BASE = "data/agent_conversation.json"
PLAN = Path("benchmark/removal_plan_v2.json")


def _cases() -> list[PurposeCase]:
    return [
        PurposeCase(
            "plan_bike_two",
            "bicicleta",
            _load(BASE, "benchmark/price_bicycle.json"),
            "c2",
            _recent("Falamos de bicicleta."),
            "Por que comprei a bicicleta e quanto custou?",
            (
                Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), ("p1",), ("trilha",)),
                Purpose("p", "preco", "bicicleta", "o_que", ("p3",), (), ("1200",)),
            ),
            "p1",
            memory_limit=180,
            split="eval",
        ),
        PurposeCase(
            "plan_two_themes",
            "misto",
            _load(BASE, "benchmark/new_bike_price.json"),
            "c1",
            _recent("Falamos do Piano V e da bicicleta."),
            "Por que escolhi o Piano V e quanto custou a bicicleta?",
            (
                Purpose("m", "motivo", "piano", "por_que", ("m6",), ("m5",), ("promocao",)),
                Purpose("p", "preco", "bicicleta", "o_que", ("b3",), (), ("1200",)),
            ),
            "m5",
            memory_limit=200,
            split="eval",
        ),
        PurposeCase(
            "plan_priority",
            "misto",
            _load(BASE, "benchmark/new_bike_price.json"),
            "c1",
            _recent("Falamos do Piano V e da bicicleta."),
            "Por que escolhi o Piano V e quanto custou a bicicleta? Priorize o motivo do Piano.",
            (
                Purpose("m", "motivo", "piano", "por_que", ("m6",), ("m5",), ("promocao",)),
                Purpose("p", "preco", "bicicleta", "o_que", ("b3",), (), ("1200",)),
            ),
            "m5",
            memory_limit=85,
            split="eval",
            priority_purpose_id="m",
        ),
        PurposeCase(
            "plan_history",
            "historico",
            _load(BASE, "benchmark/price_bicycle.json"),
            "c2",
            (RecentTurn("A bicicleta custou 1200 reais.", id="p3"), RecentTurn("Falamos de bicicleta.")),
            "Quanto custou a bicicleta?",
            (Purpose("p", "preco", "bicicleta", "o_que", ("p3",), (), ("1200",)),),
            "p1",
            split="eval",
        ),
        PurposeCase(
            "plan_restricted",
            "sensibilidade",
            _load(BASE, "benchmark/new_bike_price.json"),
            "c1",
            _recent("Falamos da bicicleta."),
            "Quanto custou a bicicleta?",
            (Purpose("p", "preco", "bicicleta", "o_que", (), (), (), unavailable=True),),
            "m5",
            split="eval",
            preferences=("Não quero falar sobre o preço da bicicleta.",),
        ),
    ]


def _render_exact(state, selected_ids: Sequence[str], removed: Sequence[str]) -> dict[str, Any]:
    by_id = {message.id: message for message in state.case.messages}
    removed_set = set(removed)
    keep = [message_id for message_id in selected_ids if message_id not in removed_set]
    evidence = [by_id[message_id] for message_id in keep if message_id in by_id and message_id != state.seed.id]
    blocks = []
    if state.seed.id in keep and state.seed.id in by_id:
        blocks.append(render_seed_block(state.seed))
    for rank, message in enumerate(evidence, start=1):
        blocks.append(render_evidence_block(message, rank))
    messages = with_memory(state.base_messages, wrap_memory(blocks))
    counter = EstimatedChatCounter(SimulatedCounter())
    used = counter.count_messages(messages)
    return {
        "messages": messages,
        "selected": ([state.seed.id] if state.seed.id in keep else []) + [message.id for message in evidence],
        "memory_tokens": used - state.base_tokens,
    }


def _verdict(
    comparison: Mapping[str, Any],
    purpose_checks: Mapping[str, Any],
    removal_checks: Mapping[str, Any],
    case: PurposeCase,
    removal: Mapping[str, Any],
) -> str:
    if comparison.get("same_request"):
        return "sem_efeito_na_requisicao"
    expected_change = removal.get("expected_change", [])
    target = next(
        (
            purpose.id
            for purpose in case.purposes
            if purpose.id in expected_change or purpose.kind in expected_change
        ),
        None,
    )
    if target is None:
        base_ok = all(value["ok"] for value in purpose_checks["purposes"].values())
        after_ok = all(value["ok"] for value in removal_checks["purposes"].values())
        return "redundante_preservado" if after_ok == base_ok else "redundante_quebrou_parte"
    base = purpose_checks["purposes"].get(target, {}).get("ok", False)
    after = removal_checks["purposes"].get(target, {}).get("ok", False)
    if not base:
        return "inconclusiva_base"
    if after:
        return "falhou"
    expected_keep = removal.get("expected_keep", [])
    preserve_ids = [
        purpose.id
        for purpose in case.purposes
        if purpose.id in expected_keep or purpose.kind in expected_keep
    ]
    preserved_ok = all(
        removal_checks["purposes"].get(purpose_id, {}).get("ok", False)
        or not purpose_checks["purposes"].get(purpose_id, {}).get("ok", False)
        for purpose_id in preserve_ids
    )
    return "confirmada" if preserved_ok else "confirmada_com_perda_colateral"


def run_removal_round(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    cases = _cases()
    train_cases, _ = purpose_training_cases()
    models = _train(train_cases, config)
    more = _train_more_model(config)
    entries = []
    planned = plan["planned_calls"]
    executed = 0
    for plan_case, case in zip(plan["cases"], cases):
        state = _initial_state(case, config)
        action = _choose_action(more, state, REFERENCE)
        current = _run_action(action, state, config)
        reduced = _reduction(state, current, config)
        purpose = _select_purpose(case, models, state, current.get("candidate_ids", []))
        reference_messages = _render_selection(state, reduced["selected"])["messages"]
        purpose_messages = _render_selection(state, purpose["selected"])["messages"]
        comparisons = []
        for removal in plan_case["removals"]:
            item = removal["id"]
            present = item in purpose["selected"] and _id_in_messages(item, purpose_messages)
            if present:
                exact = _render_exact(state, purpose["selected"], (item,))
                messages = exact["messages"]
                same = messages == purpose_messages
                selected = exact["selected"]
                memory_tokens = exact["memory_tokens"]
            else:
                messages = purpose_messages
                same = True
                selected = purpose["selected"]
                memory_tokens = purpose["memory_tokens"]
            comparisons.append(
                {
                    "item": item,
                    "role": removal["role"],
                    "expected_change": removal.get("expected_change", []),
                    "expected_keep": removal.get("expected_keep", []),
                    "present_in_purpose": present,
                    "same_request": same,
                    "messages": messages,
                    "selected": selected,
                    "memory_tokens": memory_tokens,
                }
            )
        # cache por requisição idêntica dentro do caso
        cache: dict[str, dict[str, Any]] = {}
        conditions = {"reference": reference_messages, "purpose": purpose_messages}
        for index, comparison in enumerate(comparisons, start=1):
            conditions[f"removal_{index}"] = comparison["messages"]
        results = {}
        for name, messages in conditions.items():
            key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            if key in cache:
                results[name] = dict(cache[key])
                results[name]["reused"] = True
                continue
            request = build_request_body(messages, settings)
            if client is None:
                answer_client: ChatTransport = SimulatedPurposeClient(case, messages)
            else:
                answer_client = client
            try:
                response: ChatCompletionResponse = answer_client.complete(
                    request, timeout=settings.timeout
                )
                answer = response.content
                usage = dict(response.usage)
                error = None
            except Exception as exc:
                answer = ""
                usage = {}
                error = str(exc)
            result = {
                "messages": [dict(message) for message in messages],
                "selected": (
                    purpose["selected"] if name == "purpose"
                    else reduced["selected"] if name == "reference"
                    else comparisons[int(name.split("_")[1]) - 1]["selected"]
                ),
                "answer": answer,
                "checks": _checks(case, answer),
                "token_source": "api" if usage.get("prompt_tokens") is not None else "estimativa_local",
                "usage": usage,
                "error": error,
                "reused": False,
            }
            cache[key] = result
            results[name] = result
            executed += 1
        for index, (comparison, removal) in enumerate(zip(comparisons, plan_case["removals"]), start=1):
            name = f"removal_{index}"
            verdict = _verdict(
                comparison,
                results["purpose"]["checks"],
                results[name]["checks"],
                case,
                removal,
            )
            comparison["verdict"] = verdict
            comparison["result"] = results[name]
        entries.append(
            {
                "case": case.id,
                "family": case.family,
                "conditions": results,
                "comparisons": comparisons,
            }
        )
    return {
        "planned_calls": planned,
        "executed_calls": executed,
        "cases": len(entries),
        "entries": entries,
        "human_review_pending": True,
        "existing_blind_pairs_preserved": True,
    }


def export_removal_round(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "removal_round_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, entry in enumerate(result["entries"], start=1):
        names = list(entry["conditions"])
        labels = {}
        for position, name in enumerate(names):
            labels[chr(ord("A") + position)] = entry["conditions"][name]["answer"]
        review.append(
            {
                "review_id": f"D{index:02d}",
                "case": entry["case"],
                "answers": labels,
                "rubric": "Fato solicitado, atribuição ao item, parte preservada, suporte, sensibilidade e prioridade.",
                "human_review": {
                    "reviewed": False,
                    "approved_answers": [],
                    "divergences": [],
                    "notes": "",
                },
            }
        )
    (out / "removal_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "removal_round_results.json"),
        "review": str(out / "removal_review.json"),
    }
