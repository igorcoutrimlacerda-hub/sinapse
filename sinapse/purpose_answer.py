"""Avaliação de resposta da seleção por propósito vs referência reduzida.

Inclui uma terceira condição de remoção dirigida em subconjunto pré-definido.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter
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
from .genetic_policy import REFERENCE, _choose_action as choose_action
from .memory import requires_seed
from .purpose_selection import (
    Purpose,
    PurposeCase,
    _cases as purpose_cases,
    _select_purpose,
    _train,
    _load,
    _recent,
)
from .text import normalize_text

BASE = "data/agent_conversation.json"


def _answer_terms(case: PurposeCase, purpose: Purpose) -> tuple[str, ...]:
    if purpose.answer_terms:
        return purpose.answer_terms
    mapping = {
        "quantidade": ("oito", "8", "presets"),
        "divulgacao": ("forum", "divulguei", "divulgar"),
        "motivo": ("porque", "licenca", "promocao", "trilha"),
        "correcao": ("pigments",),
        "preco": ("1200", "custou", "reais"),
    }
    return mapping.get(purpose.kind, ())


def _new_cases() -> list[PurposeCase]:
    return [
        PurposeCase(
            "new_bike_two",
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
            "new_two_themes",
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
            "new_priority",
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
            "new_history_price",
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
            "new_restricted_price",
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


def _render_selection(state, selected_ids: Sequence[str], removed: Sequence[str] = ()) -> dict[str, Any]:
    from .budget import render_evidence_block, render_seed_block, wrap_memory
    from .chat import with_memory

    by_id = {message.id: message for message in state.case.messages}
    keep = [message_id for message_id in selected_ids if message_id not in set(removed)]
    if state.seed.id not in keep and any(
        message_id != state.seed.id
        and message_id in by_id
        and requires_seed(by_id[message_id], state.seed)
        for message_id in keep
    ):
        keep.insert(0, state.seed.id)
    evidence = [by_id[message_id] for message_id in keep if message_id in by_id and message_id != state.seed.id]
    blocks = []
    if state.seed.id in keep:
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


def _checks(case: PurposeCase, answer: str) -> dict[str, Any]:
    normalized = normalize_text(answer)
    purposes = {}
    for purpose in case.purposes:
        if purpose.unavailable:
            acknowledged = any(
                marker in normalized
                for marker in (
                    "nao tenho",
                    "nao sei",
                    "sem informacao",
                    "nao consta",
                    "nao disponivel",
                    "restrito",
                    "nao encontrei",
                    "pode me lembrar",
                    "lembrar quanto",
                )
            )
            sensitive = any(term in normalized for term in ("1200", "1500", "reais"))
            purposes[purpose.id] = {
                "ok": acknowledged and not sensitive,
                "unavailable": True,
                "acknowledged": acknowledged,
                "sensitive": sensitive,
            }
            continue
        terms = _answer_terms(case, purpose)
        term_ok = any(normalize_text(term) in normalized for term in terms)
        item_ok = normalize_text(purpose.theme) in normalized if purpose.theme else True
        purposes[purpose.id] = {
            "ok": term_ok and item_ok,
            "unavailable": False,
            "term_ok": term_ok,
            "item_ok": item_ok,
        }
    priority_ok = True
    if case.priority_purpose_id:
        priority_ok = purposes.get(case.priority_purpose_id, {}).get("ok", False)
    required = [
        purpose
        for purpose in case.purposes
        if not (case.priority_purpose_id and purpose.id != case.priority_purpose_id)
    ]
    all_ok = all(purposes[purpose.id]["ok"] for purpose in required)
    count = sum(1 for purpose in case.purposes if purposes[purpose.id]["ok"])
    score = count / len(case.purposes) if case.purposes else 1.0
    if case.priority_purpose_id and not priority_ok:
        score *= 0.5
    return {
        "purposes": purposes,
        "all_ok": all_ok,
        "priority_ok": priority_ok,
        "score": round(score, 4),
        "unsupported": any(
            value.get("sensitive", False) for value in purposes.values()
        ),
    }


class SimulatedPurposeClient:
    name = "simulated-purpose"
    simulated = True

    def __init__(self, case: PurposeCase, messages: Sequence[Mapping[str, str]]) -> None:
        self.case = case
        self.messages = messages

    def complete(self, request_body: Mapping[str, Any], timeout: float = 60.0) -> ChatCompletionResponse:
        from .curiosity_learning import _id_in_messages
        parts = []
        for purpose in self.case.purposes:
            if purpose.unavailable:
                continue
            if all(_id_in_messages(message_id, self.messages) for message_id in purpose.evidence_ids):
                terms = _answer_terms(self.case, purpose)
                parts.append(f"{purpose.theme}: {'/'.join(terms)}")
        content = "; ".join(parts) if parts else "Não tenho essa informação."
        return ChatCompletionResponse(content=content, model=str(request_body.get("model", "")), finish_reason="stop", usage={}, raw={"simulated": True})


def run_purpose_answer(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    train_cases, known = purpose_cases()
    models = _train(train_cases, config)
    more = _train_more_model(config)
    cases = list(known) + _new_cases()
    removal_subset = {"new_bike_two": "m", "new_two_themes": "m"}
    entries = []
    calls = 0
    for case in cases:
        state = _initial_state(case, config)
        action = choose_action(more, state, REFERENCE)
        current = _run_action(action, state, config)
        reduced = _reduction(state, current, config)
        purpose = _select_purpose(case, models, state, current.get("candidate_ids", []))
        conditions = {
            "reference": _render_selection(state, reduced["selected"]),
            "purpose": _render_selection(state, purpose["selected"]),
        }
        target_purpose = next(
            (item for item in case.purposes if item.id == removal_subset.get(case.id)),
            None,
        )
        target_evidence = target_purpose.evidence_ids[0] if target_purpose else None
        if target_evidence is not None:
            conditions["removal"] = _render_selection(
                state, purpose["selected"], (target_evidence,)
            )
        item = {
            "case": case.id,
            "family": case.family,
            "purposes": [p.id for p in case.purposes],
            "priority": case.priority_purpose_id,
            "conditions": {},
        }
        for name, selection in conditions.items():
            calls += 1
            messages = selection["messages"]
            request = build_request_body(messages, settings)
            if client is None:
                answer_client: ChatTransport = SimulatedPurposeClient(case, messages)
            else:
                answer_client = client
            try:
                response = answer_client.complete(request, timeout=settings.timeout)
                answer = response.content
                usage = dict(response.usage)
                error = None
            except Exception as exc:
                answer = ""
                usage = {}
                error = str(exc)
            checks = _checks(case, answer)
            deps_lost = [
                message_id
                for purpose in case.purposes
                for message_id in purpose.context_ids
                if not _id_in_messages(message_id, messages)
            ]
            item["conditions"][name] = {
                "selected": selection["selected"],
                "removed": [] if name != "removal" else ([target_evidence] if target_evidence else []),
                "answer": answer,
                "checks": checks,
                "score": checks["score"],
                "dependencies_lost": deps_lost,
                "memory_tokens": selection["memory_tokens"],
                "request_tokens_estimate": EstimatedChatCounter(SimulatedCounter()).count_messages(messages),
                "token_source": "api" if usage.get("prompt_tokens") is not None else "estimativa_local",
                "usage": usage,
                "error": error,
            }
        entries.append(item)
    summary = {
        name: _aggregate(entries, name)
        for name in ("reference", "purpose", "removal")
    }
    pairs = {}
    for name in ("purpose", "removal"):
        subset = [entry for entry in entries if name in entry["conditions"]]
        pairs[name] = {
            "better_than_reference": sum(
                1
                for entry in subset
                if entry["conditions"][name]["score"] > entry["conditions"]["reference"]["score"]
            ),
            "worse_than_reference": sum(
                1
                for entry in subset
                if entry["conditions"][name]["score"] < entry["conditions"]["reference"]["score"]
            ),
            "tied_with_reference": sum(
                1
                for entry in subset
                if entry["conditions"][name]["score"] == entry["conditions"]["reference"]["score"]
            ),
        }
    return {
        "cases": len(entries),
        "calls": calls,
        "conditions": summary,
        "pairs": pairs,
        "entries": entries,
        "removal_subset": removal_subset,
    }


def _aggregate(entries: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    items = [entry["conditions"][name] for entry in entries if name in entry["conditions"]]
    return {
        "cases": len(items),
        "all_ok": sum(1 for item in items if item["checks"]["all_ok"]),
        "purposes_ok": sum(
            1
            for item in items
            for value in item["checks"]["purposes"].values()
            if value["ok"]
        ),
        "purposes_total": sum(len(item["checks"]["purposes"]) for item in items),
        "unsupported": sum(1 for item in items if item["checks"]["unsupported"]),
        "sensitive": sum(
            1
            for item in items
            for value in item["checks"]["purposes"].values()
            if value.get("sensitive", False)
        ),
        "dependencies_lost": sum(len(item["dependencies_lost"]) for item in items),
        "priority_ok": sum(1 for item in items if item["checks"]["priority_ok"]),
        "avg_score": round(sum(item["score"] for item in items) / len(items), 4),
        "avg_memory_tokens": round(
            sum(item["memory_tokens"] for item in items) / len(items), 1
        ),
        "prompt_tokens": sum(
            int(item["usage"].get("prompt_tokens", 0) or 0) for item in items
        ),
        "completion_tokens": sum(
            int(item["usage"].get("completion_tokens", 0) or 0) for item in items
        ),
    }


def export_purpose_answer(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "purpose_answer_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, entry in enumerate(result["entries"], start=1):
        labels = list(entry["conditions"].keys())
        rotated = labels[index % len(labels):] + labels[:index % len(labels)]
        review.append(
            {
                "review_id": f"P{index:02d}",
                "case": entry["case"],
                "answers": {
                    chr(ord("A") + position): entry["conditions"][name]["answer"]
                    for position, name in enumerate(rotated)
                },
                "rubric": "Fatos por propósito, atribuição, suporte, sensibilidade, prioridade e partes indisponíveis.",
            }
        )
    (out / "purpose_answer_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "purpose_answer_results.json"),
        "review": str(out / "purpose_answer_review.json"),
    }
