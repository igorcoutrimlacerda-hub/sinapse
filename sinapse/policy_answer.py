"""Comparação das três políticas congeladas com respostas e revisão cega."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .genetic_policy import Policy, _choose_action, _initial_state, _reduction, run_genetic_experiment
from .context_reduction import _run_action, _train_more_model
from .paired_answer import (
    META,
    SimulatedPairClient,
    _answer_checks,
    _cases as known_answer_cases,
)
from .chat import EstimatedChatCounter
from .budget import SimulatedCounter


def run_policy_answer(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    experiment = run_genetic_experiment(config)
    policies = {
        name: Policy(**values)
        for name, values in experiment["frozen_policies"].items()
    }
    model = _train_more_model(config)
    cases = known_answer_cases()
    counter = EstimatedChatCounter(SimulatedCounter())
    entries = []
    calls = 0
    for case in cases:
        meta = META[case.id]
        state = _initial_state(case, config)
        results = {}
        for name, policy in policies.items():
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
            messages = reduced["messages"]
            request = build_request_body(messages, settings)
            calls += 1
            if client is None:
                answer_client: ChatTransport = SimulatedPairClient(meta, messages)
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
            checks = _answer_checks(meta, answer)
            results[name] = {
                "policy": policy.as_dict(),
                "action": action,
                "selected": reduced["selected"],
                "removed": list(reduced["removed"]),
                "messages": [dict(message) for message in messages],
                "answer": answer,
                "checks": checks,
                "score": checks["score"],
                "memory_tokens": reduced["memory_tokens"],
                "request_tokens_estimate": counter.count_messages(messages),
                "token_source": "api"
                if usage.get("prompt_tokens") is not None
                else "estimativa_local",
                "usage": usage,
                "error": error,
            }
        entries.append({"case": case.id, "family": case.family, "results": results})
    summary = {
        name: _aggregate(entries, name) for name in policies
    }
    comparisons = {
        name: {
            "better_than_reference": sum(
                1
                for entry in entries
                if entry["results"][name]["score"]
                > entry["results"]["reference"]["score"]
            ),
            "worse_than_reference": sum(
                1
                for entry in entries
                if entry["results"][name]["score"]
                < entry["results"]["reference"]["score"]
            ),
            "tied_with_reference": sum(
                1
                for entry in entries
                if entry["results"][name]["score"]
                == entry["results"]["reference"]["score"]
            ),
        }
        for name in policies
    }
    return {
        "cases": len(entries),
        "calls": calls,
        "policies": {name: policy.as_dict() for name, policy in policies.items()},
        "summary": summary,
        "comparisons": comparisons,
        "entries": entries,
    }


def _aggregate(entries: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    items = [entry["results"][name] for entry in entries]
    return {
        "cases": len(items),
        "facts": sum(1 for item in items if item["checks"]["facts_ok"]),
        "item": sum(1 for item in items if item["checks"]["item_ok"]),
        "unsupported": sum(1 for item in items if item["checks"]["unsupported"]),
        "sensitive": sum(1 for item in items if item["checks"]["sensitive"]),
        "ambiguity": sum(1 for item in items if item["checks"]["ambiguity_ok"]),
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


def export_policy_answer(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    policy_names = list(result["policies"])
    for index, entry in enumerate(result["entries"], start=1):
        offset = (index - 1) % len(policy_names)
        rotated = policy_names[offset:] + policy_names[:offset]
        review.append(
            {
                "review_id": f"C{index:02d}",
                "case": entry["case"],
                "answers": {
                    chr(ord("A") + position): entry["results"][name]["answer"]
                    for position, name in enumerate(rotated)
                },
                "rubric": "Compare fatos, atribuição, suporte, sensibilidade e ambiguidade; não sabe qual política gerou A, B ou C.",
            }
        )
    (out / "policy_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "policy_results.json"),
        "review": str(out / "policy_review.json"),
    }
