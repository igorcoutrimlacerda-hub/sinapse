"""Consolidação da revisão cega do experimento de instrução.

Só revela o vínculo A/B com as instruções depois que todas as notas humanas
aplicáveis estiverem registradas. A rubrica v2 e os resultados automáticos
permanecem em arquivos separados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

INSTRUCTIONS = ("current", "experimental")


def _missing(pair: Mapping[str, Any]) -> list[str]:
    missing = []
    if not pair.get("human_review", {}).get("reviewed"):
        missing.append(f"{pair['review_id']}:human_review.reviewed")
    if pair["human_review"].get("preferred") not in ("A", "B", "empate"):
        missing.append(f"{pair['review_id']}:human_review.preferred")
    for label in ("A", "B"):
        for name in pair["criteria"]:
            field = pair.get("human", {}).get(label, {}).get(name, {})
            if field.get("score") not in (0, 1):
                missing.append(f"{pair['review_id']}:{label}:{name}:score")
    return missing


def _row_for(
    pair: Mapping[str, Any],
    label: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["case"] == pair["case"]
        and row["answer"] == pair["answers"][label]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{pair['review_id']}: resposta {label} com {len(matches)} "
            "correspondências"
        )
    return matches[0]


def _criteria_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("decision", "reason", "completed", "pending", "condition")
    result = {}
    for name in names:
        applicable = [row for row in rows if name in row["criteria"]]
        result[name] = {
            "hits": sum(row["human"][name] for row in applicable),
            "total": len(applicable),
        }
    return result


def _v2_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("decision", "reason", "completed", "pending", "condition")
    result = {}
    for name in names:
        applicable = [row for row in rows if name in row["criteria"]]
        result[name] = {
            "hits": sum(int(row["v2"][name]) for row in applicable),
            "total": len(applicable),
        }
    return result


def consolidate_instruction_review(
    review_path: str | Path = "benchmark_out_instruction_experiment/instruction_review_blind.json",
    results_path: str | Path = "benchmark_out_instruction_experiment/instruction_results.json",
    out_path: str | Path = "benchmark_out_instruction_experiment/instruction_consolidated.json",
) -> dict[str, Any]:
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    missing = [item for pair in review for item in _missing(pair)]
    if missing:
        return {
            "complete": False,
            "revealed": False,
            "missing": missing,
            "note": "Condições ocultas até o registro de todas as notas.",
        }
    rows = results["rows"]
    by_instruction: dict[str, list[dict[str, Any]]] = {
        name: [] for name in INSTRUCTIONS
    }
    revealed = []
    divergences = []
    preferences = {name: {"wins": 0, "ties": 0} for name in INSTRUCTIONS}
    for pair in review:
        mapping = {}
        for label in ("A", "B"):
            row = _row_for(pair, label, rows)
            condition = row["instruction"]
            mapping[label] = condition
            human = {
                name: pair["human"][label][name]["score"]
                for name in pair["criteria"]
            }
            v2 = {
                name: bool(row["v2"]["checks"][name])
                for name in pair["criteria"]
            }
            by_instruction[condition].append(
                {
                    "review_id": pair["review_id"],
                    "case": pair["case"],
                    "human": human,
                    "v2": v2,
                    "criteria": list(pair["criteria"]),
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "memory_tokens": row["memory_tokens"],
                    "v2_complete": row["v2"]["complete"],
                }
            )
            for name in pair["criteria"]:
                if human[name] != int(v2[name]):
                    divergences.append(
                        {
                            "review_id": pair["review_id"],
                            "case": pair["case"],
                            "condition": condition,
                            "criterion": name,
                            "human": human[name],
                            "v2": int(v2[name]),
                        }
                    )
        if len(set(mapping.values())) != 2:
            raise ValueError(f"{pair['review_id']}: A/B na mesma instrução")
        preferred = pair["human_review"]["preferred"]
        if preferred in mapping:
            preferences[mapping[preferred]]["wins"] += 1
        elif preferred == "empate":
            for condition in mapping.values():
                preferences[condition]["ties"] += 1
        revealed.append(
            {
                "review_id": pair["review_id"],
                "case": pair["case"],
                "conditions": mapping,
                "preferred": preferred,
            }
        )
    summary = {}
    for condition, group in by_instruction.items():
        summary[condition] = {
            "rows": len(group),
            "human_criteria": _criteria_counts(group),
            "v2_criteria": _v2_counts(group),
            "human_complete": sum(
                all(row["human"][name] == 1 for name in row["human"])
                for row in group
            ),
            "v2_complete": sum(row["v2_complete"] for row in group),
            "prompt_tokens": sum(row["prompt_tokens"] for row in group),
            "completion_tokens": sum(
                row["completion_tokens"] for row in group
            ),
            "memory_tokens_avg": round(
                sum(row["memory_tokens"] for row in group) / len(group), 1
            ),
        }
    result = {
        "complete": True,
        "revealed": True,
        "reviewed_pairs": len(review),
        "review_method": (
            "revisão cega assistida por IA; rótulos A/B ocultos até a "
            "consolidação"
        ),
        "revealed_conditions": revealed,
        "preferences": preferences,
        "human_summary": summary,
        "divergences": divergences,
        "divergences_by_criterion": {
            name: sum(1 for item in divergences if item["criterion"] == name)
            for name in ("decision", "reason", "completed", "pending", "condition")
        },
        "automatic_preserved": str(results_path),
        "instruction_cost": results["instruction_cost"],
        "cost": {
            condition: {
                "prompt_tokens": summary[condition]["prompt_tokens"],
                "completion_tokens": summary[condition]["completion_tokens"],
                "memory_tokens_avg": summary[condition]["memory_tokens_avg"],
            }
            for condition in INSTRUCTIONS
        },
        "cost_delta_prompt": (
            summary["experimental"]["prompt_tokens"]
            - summary["current"]["prompt_tokens"]
        ),
        "cost_delta_completion": (
            summary["experimental"]["completion_tokens"]
            - summary["current"]["completion_tokens"]
        ),
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
