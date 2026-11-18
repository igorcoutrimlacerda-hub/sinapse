"""Reprocessa resultados salvos por propósito, sem novas chamadas.

Separa aprovação da rubrica de cobertura efetiva de todos os propósitos,
identifica omissões corretas, agrupa regressões e famílias novas e prepara
revisão humana cega.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

KNOWN = {
    "ev_bike_motive",
    "ev_piano_motive",
    "ev_pack_both",
    "ev_two_themes",
    "ev_serum_contrast",
    "ev_history_bike",
    "ev_restricted",
}

RUBRIC = {
    "por_proposito": "fato correto e atribuição ao item certo",
    "prioridade": "prioridade explícita deve ser respondida; parte secundária pode ficar pendente se reconhecida",
    "indisponivel": "parte sem evidência ou restrita deve ser reconhecida sem invenção",
    "sensibilidade": "não revelar conteúdo restrito",
}


def _case_metrics(entry: Mapping[str, Any], condition: str) -> dict[str, Any] | None:
    if condition not in entry["conditions"]:
        return None
    item = entry["conditions"][condition]
    purposes = item["checks"]["purposes"]
    return {
        "rubric_approved": bool(item["checks"]["all_ok"]),
        "all_purposes": all(value["ok"] for value in purposes.values()) if purposes else True,
        "purposes_ok": sum(1 for value in purposes.values() if value["ok"]),
        "purposes_total": len(purposes),
        "prompt": int(item["usage"].get("prompt_tokens", 0) or 0),
        "completion": int(item["usage"].get("completion_tokens", 0) or 0),
        "score": item["score"],
    }


def reprocess(results_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    groups = {"regression": [], "new": []}
    for entry in data["entries"]:
        group = "regression" if entry["case"] in KNOWN else "new"
        groups[group].append(entry)

    summary: dict[str, Any] = {"groups": {}, "overall": {}}
    for group, entries in groups.items():
        group_summary = {}
        for condition in ("reference", "purpose", "removal"):
            metrics = [
                value for value in (_case_metrics(entry, condition) for entry in entries)
                if value is not None
            ]
            if not metrics:
                continue
            group_summary[condition] = {
                "cases": len(metrics),
                "rubric_approved": sum(item["rubric_approved"] for item in metrics),
                "all_purposes": sum(item["all_purposes"] for item in metrics),
                "purposes_ok": sum(item["purposes_ok"] for item in metrics),
                "purposes_total": sum(item["purposes_total"] for item in metrics),
                "prompt": sum(item["prompt"] for item in metrics),
                "completion": sum(item["completion"] for item in metrics),
                "delta_prompt_vs_reference": sum(
                    item["prompt"] for item in metrics
                )
                - sum(
                    value["prompt"]
                    for value in (
                        _case_metrics(entry, "reference") for entry in entries
                    )
                    if value is not None
                ),
                "delta_completion_vs_reference": sum(
                    item["completion"] for item in metrics
                )
                - sum(
                    value["completion"]
                    for value in (
                        _case_metrics(entry, "reference") for entry in entries
                    )
                    if value is not None
                ),
            }
        paired = {"purpose": {"better": 0, "tied": 0, "worse": 0}}
        for entry in entries:
            ref = _case_metrics(entry, "reference")
            purpose = _case_metrics(entry, "purpose")
            if ref is None or purpose is None:
                continue
            if purpose["score"] > ref["score"]:
                paired["purpose"]["better"] += 1
            elif purpose["score"] < ref["score"]:
                paired["purpose"]["worse"] += 1
            else:
                paired["purpose"]["tied"] += 1
        group_summary["paired"] = paired
        summary["groups"][group] = group_summary

    overall = {}
    for condition in ("reference", "purpose", "removal"):
        metrics = [
            value for value in (_case_metrics(entry, condition) for entry in data["entries"])
            if value is not None
        ]
        overall[condition] = {
            "cases": len(metrics),
            "rubric_approved": sum(item["rubric_approved"] for item in metrics),
            "all_purposes": sum(item["all_purposes"] for item in metrics),
            "purposes_ok": sum(item["purposes_ok"] for item in metrics),
            "purposes_total": sum(item["purposes_total"] for item in metrics),
            "prompt": sum(item["prompt"] for item in metrics),
            "completion": sum(item["completion"] for item in metrics),
        }
    summary["overall"] = overall

    omissions = []
    accepted_unavailable = []
    priority_case = next(
        (entry for entry in data["entries"] if entry["case"] == "new_priority"), None
    )
    for entry in data["entries"]:
        for condition in ("reference", "purpose"):
            item = entry["conditions"].get(condition)
            if item is None:
                continue
            for purpose_id, check in item["checks"]["purposes"].items():
                if check.get("unavailable") and check["ok"]:
                    accepted_unavailable.append(
                        {
                            "case": entry["case"],
                            "condition": condition,
                            "purpose": purpose_id,
                            "reason": "restricao_indisponivel",
                        }
                    )
                if check["ok"]:
                    continue
                reason = "nao_respondido"
                if (
                    entry.get("priority")
                    and purpose_id != entry["priority"]
                    and item["checks"].get("priority_ok")
                ):
                    reason = "prioridade_explicita"
                elif check.get("unavailable"):
                    reason = "restricao_indisponivel"
                elif not any(
                    evidence in item["selected"]
                    for evidence in entry.get("expected_ids", [])
                ):
                    reason = "evidencia_nao_enviada"
                omissions.append(
                    {
                        "case": entry["case"],
                        "condition": condition,
                        "purpose": purpose_id,
                        "reason": reason,
                    }
                )
    summary["omissions"] = omissions
    summary["accepted_unavailable"] = accepted_unavailable
    summary["priority_case"] = priority_case
    summary["entries"] = data["entries"]
    summary["calls"] = data.get("calls", 0)
    return summary


def export_human_review(review_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    output = []
    for entry in review:
        output.append(
            {
                **entry,
                "group": "regression" if entry["case"] in KNOWN else "new",
                "rubric_frozen": RUBRIC,
                "human_review": {
                    "reviewed": False,
                    "approved_answers": [],
                    "divergences": [],
                    "notes": "",
                },
            }
        )
    Path(out_path).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"pairs": len(output), "pending": sum(1 for entry in output if not entry["human_review"]["reviewed"])}
