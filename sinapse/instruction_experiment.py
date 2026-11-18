"""Experimento de instrução de geração com memória e orçamento fixos.

Usa os dez casos do plano, a mesma memória multi-propósito e o mesmo histórico;
apenas a instrução do sistema varia. Pontua com a rubrica v2 congelada e
exporta respostas A/B sem rótulo de instrução.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import SimulatedCounter
from .config import load_config
from .deepseek import ChatCompletionResponse, ChatTransport, build_request_body
from .models import load_messages
from .resumption import _multipurpose_context, cases
from .resumption_rubric_v2 import evaluate_v2, specs
from .text import normalize_text

PLAN = "benchmark_out_resumption_real/instruction_experiment_plan.json"
DATASET = "benchmark/resumption_sessions.json"
INSTRUCTIONS = ("current", "experimental")


def _memory_text(messages: Sequence[Mapping[str, str]], base_instruction: str) -> str:
    system = str(messages[0]["content"])
    prefix = base_instruction.strip()
    if system.startswith(prefix):
        return system[len(prefix) :].lstrip("\n")
    return system


def _messages_with_instruction(
    messages: Sequence[Mapping[str, str]],
    base_instruction: str,
    instruction: str,
) -> tuple[dict[str, str], ...]:
    memory = _memory_text(messages, base_instruction)
    system = instruction.strip()
    if memory:
        system += "\n\n" + memory
    return ({"role": "system", "content": system},) + tuple(
        dict(message) for message in messages[1:]
    )


def _basis(
    messages: Sequence[Mapping[str, str]], base_instruction: str
) -> dict[str, Any]:
    return {
        "memory": _memory_text(messages, base_instruction),
        "non_system": [dict(message) for message in messages[1:]],
    }


def _blind_order(case: str, instruction: str) -> str:
    return hashlib.sha256(f"{case}:{instruction}".encode()).hexdigest()


def _criteria(spec) -> list[str]:
    names = list(spec.components)
    if spec.condition_facts:
        names.append("condition")
    return names


def _v2(spec, answer: str) -> dict[str, Any]:
    score, checks = evaluate_v2(spec, answer)
    failed = [name for name in _criteria(spec) if not checks[name]]
    return {
        "score": score,
        "checks": checks,
        "complete": not failed,
        "failed": failed,
    }


def _rp02(answer: str, checks: Mapping[str, Any]) -> dict[str, Any]:
    text = normalize_text(answer)
    return {
        "test_intro_96": bool(checks["pending"]),
        "cpu_goal": "cpu" in text
        and any(term in text for term in ("estoura", "medir", "verificar", "confirmar")),
        "unsupported_additions": [
            term for term in ("clipping", "dropouts") if term in text
        ],
    }


def _dunas(answer: str, checks: Mapping[str, Any]) -> dict[str, Any]:
    text = normalize_text(answer)
    return {
        "threshold_30": "30" in text,
        "export_path": "exportar" in text,
        "fallback_path": any(term in text for term in ("oversampling", "reduz")),
        "condition": bool(checks["condition"]),
    }


def run_instruction_experiment(
    config: Mapping[str, Any] | None = None,
    settings=None,
    client: ChatTransport | None = None,
    plan_path: str | Path = PLAN,
    dataset: str | Path = DATASET,
) -> dict[str, Any]:
    if client is None:
        raise ValueError("Um cliente real é obrigatório")
    from .deepseek import load_deepseek_settings

    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    instructions = plan["instructions"]
    current_instruction = str(instructions["current"])
    experimental_instruction = str(instructions["experimental"])
    base_instruction = str(config.get("instructions", ""))
    corpus = load_messages(dataset)
    case_by_id = {case.id: case for case in cases()}
    spec_by_case = specs()
    rows = []
    calls = 0
    for case_id in plan["cases"]:
        case = case_by_id[case_id]
        memory_corpus = [
            message for message in corpus if message.id not in case.recent_ids
        ]
        messages, selected, memory_tokens, _ = _multipurpose_context(
            case, memory_corpus, config, settings
        )
        variants = {
            "current": _messages_with_instruction(
                messages, base_instruction, current_instruction
            ),
            "experimental": _messages_with_instruction(
                messages, base_instruction, experimental_instruction
            ),
        }
        bases = {
            name: _basis(variant, instruction)
            for name, variant, instruction in (
                ("current", variants["current"], current_instruction),
                ("experimental", variants["experimental"], experimental_instruction),
            )
        }
        invariant = bases["current"] == bases["experimental"]
        if not invariant:
            raise RuntimeError(f"{case_id}: contexto diverge entre instruções")
        requests = {
            name: build_request_body(variant, settings)
            for name, variant in variants.items()
        }
        fixed = {
            key: value
            for key, value in requests["current"].items()
            if key != "messages"
        }
        fixed_experimental = {
            key: value
            for key, value in requests["experimental"].items()
            if key != "messages"
        }
        if fixed != fixed_experimental:
            raise RuntimeError(f"{case_id}: parâmetros de geração divergem")
        for name in INSTRUCTIONS:
            request = requests[name]
            started = time.perf_counter()
            error = None
            answer = ""
            usage: Mapping[str, Any] = {}
            try:
                response: ChatCompletionResponse = client.complete(
                    request, timeout=settings.timeout
                )
                answer = response.content
                usage = response.usage
            except Exception as exc:  # pragma: no cover - transporte real
                error = str(exc)
            latency = (time.perf_counter() - started) * 1000
            calls += 1
            scored = _v2(spec_by_case[case.id], answer)
            rows.append(
                {
                    "case": case.id,
                    "instruction": name,
                    "selected": list(selected),
                    "memory_tokens": memory_tokens,
                    "answer": answer,
                    "error": error,
                    "latency_ms": round(latency, 2),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        usage.get("completion_tokens", 0) or 0
                    ),
                    "token_source": "api" if usage else "indisponivel",
                    "v2": scored,
                }
            )
    # Agregados por instrução.
    aggregates = {}
    for name in INSTRUCTIONS:
        group = [row for row in rows if row["instruction"] == name]
        criteria = {}
        for criterion in ("decision", "reason", "completed", "pending", "condition"):
            applicable = [
                row
                for row in group
                if criterion in row["v2"]["checks"]["components"]
                or (
                    criterion == "condition"
                    and "condition" in row["v2"]["checks"]
                    and row["case"] == "proximo_condicionado"
                )
            ]
            criteria[criterion] = {
                "hits": sum(row["v2"]["checks"][criterion] for row in applicable),
                "total": len(applicable),
            }
        aggregates[name] = {
            "rows": len(group),
            "complete": sum(row["v2"]["complete"] for row in group),
            "criteria": criteria,
            "prompt_tokens": sum(row["prompt_tokens"] for row in group),
            "completion_tokens": sum(
                row["completion_tokens"] for row in group
            ),
            "memory_tokens_avg": round(
                sum(row["memory_tokens"] for row in group) / len(group), 1
            ),
            "errors": sum(1 for row in group if row["error"]),
        }
    pairs = []
    for case_id in plan["cases"]:
        current = next(
            row
            for row in rows
            if row["case"] == case_id and row["instruction"] == "current"
        )
        experimental = next(
            row
            for row in rows
            if row["case"] == case_id and row["instruction"] == "experimental"
        )
        delta = round(experimental["v2"]["score"] - current["v2"]["score"], 4)
        pairs.append(
            {
                "case": case_id,
                "current_score": current["v2"]["score"],
                "experimental_score": experimental["v2"]["score"],
                "current_complete": current["v2"]["complete"],
                "experimental_complete": experimental["v2"]["complete"],
                "delta": delta,
                "improved": delta > 0,
                "tied": delta == 0,
                "worse": delta < 0,
            }
        )
    rp02 = {
        row["instruction"]: _rp02(row["answer"], row["v2"]["checks"])
        for row in rows
        if row["case"] == "proximo_teste"
    }
    dunas = {
        row["instruction"]: _dunas(row["answer"], row["v2"]["checks"])
        for row in rows
        if row["case"] == "proximo_condicionado"
    }
    blind = []
    for index, case_id in enumerate(plan["cases"], start=1):
        group = [row for row in rows if row["case"] == case_id]
        ordered = sorted(group, key=lambda row: _blind_order(case_id, row["instruction"]))
        criteria = _criteria(spec_by_case[case_id])
        blank = {
            name: {"score": None, "notes": ""} for name in criteria
        }
        blind.append(
            {
                "review_id": f"IE{index:02d}",
                "case": case_id,
                "question": case_by_id[case_id].question,
                "rubric": case_by_id[case_id].rubric,
                "criteria": criteria,
                "answers": {
                    "A": ordered[0]["answer"],
                    "B": ordered[1]["answer"],
                },
                "human": {"A": dict(blank), "B": dict(blank)},
                "human_review": {
                    "reviewed": False,
                    "preferred": None,
                    "notes": "",
                },
            }
        )
    instruction_cost = {
        "current_instruction_tokens": SimulatedCounter().count(current_instruction),
        "experimental_instruction_tokens": SimulatedCounter().count(
            experimental_instruction
        ),
        "estimated_delta": SimulatedCounter().count(experimental_instruction)
        - SimulatedCounter().count(current_instruction),
        "real_prompt_delta_total": sum(
            row["prompt_tokens"]
            for row in rows
            if row["instruction"] == "experimental"
        )
        - sum(
            row["prompt_tokens"]
            for row in rows
            if row["instruction"] == "current"
        ),
    }
    errors = {
        "new_experimental": [
            {
                "case": pair["case"],
                "failed": next(
                    row["v2"]["failed"]
                    for row in rows
                    if row["case"] == pair["case"]
                    and row["instruction"] == "experimental"
                ),
            }
            for pair in pairs
            if pair["worse"]
        ],
        "new_current": [
            {
                "case": pair["case"],
                "failed": next(
                    row["v2"]["failed"]
                    for row in rows
                    if row["case"] == pair["case"]
                    and row["instruction"] == "current"
                ),
            }
            for pair in pairs
            if pair["current_complete"] is False
        ],
        "persistent": [
            {
                "case": pair["case"],
                "failed": sorted(
                    set(
                        next(
                            row["v2"]["failed"]
                            for row in rows
                            if row["case"] == pair["case"]
                            and row["instruction"] == "current"
                        )
                    )
                    & set(
                        next(
                            row["v2"]["failed"]
                            for row in rows
                            if row["case"] == pair["case"]
                            and row["instruction"] == "experimental"
                        )
                    )
                ),
            }
            for pair in pairs
            if not pair["current_complete"]
            and not pair["experimental_complete"]
        ],
    }
    return {
        "plan": str(plan_path),
        "mode": "real",
        "model": settings.model,
        "calls": calls,
        "pairs": len(plan["cases"]),
        "rows": rows,
        "aggregates": aggregates,
        "pairs_comparison": pairs,
        "improved": sum(pair["improved"] for pair in pairs),
        "tied": sum(pair["tied"] for pair in pairs),
        "worse": sum(pair["worse"] for pair in pairs),
        "rp02": rp02,
        "dunas": dunas,
        "instruction_cost": instruction_cost,
        "errors": errors,
        "blind_review": blind,
        "known_scenarios": True,
        "real_tokens": {
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(
                row["completion_tokens"] for row in rows
            ),
            "source": "api",
        },
    }


def export_instruction_experiment(
    result: Mapping[str, Any], out_dir: str | Path
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instruction_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "instruction_review_blind.json").write_text(
        json.dumps(result["blind_review"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "EXPERIMENTO DE INSTRUÇÃO",
        f"modelo={result['model']} | pares={result['pairs']} | chamadas={result['calls']}",
        f"tokens reais: prompt={result['real_tokens']['prompt_tokens']} "
        f"completion={result['real_tokens']['completion_tokens']}",
        f"custo da instrução: {result['instruction_cost']}",
    ]
    for name, item in result["aggregates"].items():
        criteria = " ".join(
            f"{criterion}={data['hits']}/{data['total']}"
            for criterion, data in item["criteria"].items()
            if data["total"]
        )
        lines.append(
            f"{name:<14} completos={item['complete']}/{item['rows']} "
            f"{criteria} prompt={item['prompt_tokens']} "
            f"completion={item['completion_tokens']} "
            f"memória_média={item['memory_tokens_avg']} erros={item['errors']}"
        )
    lines.append(
        f"pares: melhoraram={result['improved']} empataram={result['tied']} "
        f"pioraram={result['worse']}"
    )
    lines.append(f"RP02: {result['rp02']}")
    lines.append(f"Dunas: {result['dunas']}")
    lines.append(f"erros: {result['errors']}")
    (out / "instruction_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "results": str(out / "instruction_results.json"),
        "review": str(out / "instruction_review_blind.json"),
        "summary": str(out / "instruction_summary.txt"),
    }
