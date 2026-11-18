"""Execução pareada real dos dez pares de retomada, preservando o simulado.

Usa exatamente os pares congelados em ``real_comparison.json``, reconstrói as
duas políticas congeladas, envia cada condição ao transporte real e grava
resultados e revisão cega em um diretório separado.
"""

from __future__ import annotations

import hashlib
import json
import time
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
from .models import Message, load_messages
from .resumption import (
    _memory_ids,
    _multipurpose_context,
    _run_agent,
    cases,
)
from .resumption_validation import _validate_condition
from .text import normalize_text

REAL_CONDITIONS = ("sinapse", "sinapse_multipurpose")
DATASET = "benchmark/resumption_sessions.json"


def _contexts(
    case,
    corpus: Sequence[Message],
    config: Mapping[str, Any],
    settings: DeepSeekSettings,
):
    memory_corpus = [
        message for message in corpus if message.id not in case.recent_ids
    ]
    single_turn = _run_agent(case, memory_corpus, config, settings)
    single = (single_turn.messages, _memory_ids(single_turn))
    if case.open_resumption:
        messages, selected, _, _ = _multipurpose_context(
            case, memory_corpus, config, settings
        )
        multi = (messages, selected)
    else:
        multi = single
    return {"sinapse": single, "sinapse_multipurpose": multi}


def summarize_real(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_case = {case.id: case for case in cases()}
    aggregates = {}
    for condition in REAL_CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        item: dict[str, Any] = {
            "rows": len(group),
            "complete": sum(row["complete"] for row in group),
            "complete_diagnostic": sum(
                row["complete_diagnostic"] for row in group
            ),
            "reference_failures": sum(
                len(row["reference_failures"]) for row in group
            ),
            "reference_support": sum(
                not row["reference_failures"] for row in group
            ),
            "prompt_tokens": sum(row["prompt_tokens"] for row in group),
            "completion_tokens": sum(
                row["completion_tokens"] for row in group
            ),
            "errors": sum(1 for row in group if row["error"]),
        }
        for name in ("decision", "reason", "completed", "pending"):
            subset = [
                row
                for row in group
                if name in row["checks"]["components"]
            ]
            item[f"{name}_ok"] = sum(
                row["checks"][name] for row in subset
            )
            item[f"{name}_total"] = len(subset)
        item["decision_fact_only"] = sum(
            row["decision_fact_present"] for row in group
        )
        condition_rows = [
            row
            for row in group
            if by_case[row["case"]].condition_ids
        ]
        item["conditions_ok"] = sum(
            row["conditions_ok"] for row in condition_rows
        )
        item["conditions_total"] = len(condition_rows)
        project_rows = [
            row
            for row in group
            if by_case[row["case"]].wrong_project_ids
        ]
        item["project_ok"] = sum(
            row["project_ok"] for row in project_rows
        )
        item["project_total"] = len(project_rows)
        aggregates[condition] = item
    return aggregates


def run_real_pairs(
    pairs_path: str | Path,
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
    dataset: str | Path = DATASET,
) -> dict[str, Any]:
    if client is None:
        raise ValueError("Um cliente real é obrigatório para esta execução")
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))["pairs"]
    corpus = load_messages(dataset)
    by_case = {case.id: case for case in cases()}
    rows = []
    calls = 0
    for pair in pairs:
        case = by_case[pair["case"]]
        contexts = _contexts(case, corpus, config, settings)
        for condition in REAL_CONDITIONS:
            messages, selected = contexts[condition]
            request = build_request_body(messages, settings)
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
            validation = _validate_condition(
                case, condition, selected, messages, answer, corpus
            )
            normalized = normalize_text(answer)
            decision_fact_present = (
                any(
                    normalize_text(fact) in normalized
                    for fact in case.decision_facts
                )
                if case.decision_facts
                else False
            )
            diagnostic_checks = dict(validation["checks"])
            if "decision" in diagnostic_checks:
                diagnostic_checks["decision"] = decision_fact_present
            complete_diagnostic = all(
                diagnostic_checks[name]
                for name in diagnostic_checks["components"]
            )
            rows.append(
                {
                    "case": case.id,
                    "condition": condition,
                    "question": case.question,
                    "selected": list(selected),
                    "answer": answer,
                    "error": error,
                    "latency_ms": round(latency, 2),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        usage.get("completion_tokens", 0) or 0
                    ),
                    "token_source": (
                        "api"
                        if usage.get("prompt_tokens") is not None
                        else "indisponivel"
                    ),
                    "complete": validation["complete"],
                    "complete_diagnostic": complete_diagnostic,
                    "decision_fact_present": decision_fact_present,
                    "checks": validation["checks"],
                    "missing_elements": validation["missing_elements"],
                    "reference_failures": validation["reference_failures"],
                    "reference_support": validation["reference_support"],
                    "conditions_ok": validation["conditions_ok"],
                    "conditions_omitted": validation["conditions_omitted"],
                    "condition_paths": validation["condition_paths"],
                    "project_ok": validation["project_ok"],
                    "wrong_project_selected": validation[
                        "wrong_project_selected"
                    ],
                    "wrong_project_terms": validation["wrong_project_terms"],
                }
            )
    aggregates = summarize_real(rows)
    blind = []
    for index, pair in enumerate(pairs, start=1):
        group = [row for row in rows if row["case"] == pair["case"]]
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{row['case']}:{row['condition']}".encode()
            ).hexdigest(),
        )
        blind.append(
            {
                "review_id": f"RP{index:02d}",
                "case": pair["case"],
                "question": pair["question"],
                "rubric": pair["rubric"],
                "answers": {
                    "A": ordered[0]["answer"],
                    "B": ordered[1]["answer"],
                },
                "human_review": {
                    "reviewed": False,
                    "preferred": None,
                    "notes": "",
                },
            }
        )
    return {
        "pairs_file": str(pairs_path),
        "mode": "real",
        "model": settings.model,
        "calls": calls,
        "pairs": len(pairs),
        "rows": rows,
        "aggregates": aggregates,
        "blind_review": blind,
        "real_tokens": {
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(
                row["completion_tokens"] for row in rows
            ),
            "source": "api",
        },
    }


def export_real_pairs(
    result: Mapping[str, Any], out_dir: str | Path
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "real_review_blind.json").write_text(
        json.dumps(result["blind_review"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    aggregates = result["aggregates"]
    lines = [
        "RETOMADA REAL — PARES PAREADOS",
        f"modelo={result['model']} | pares={result['pairs']} | "
        f"chamadas={result['calls']}",
    ]
    for condition, item in aggregates.items():
        lines.append(
            f"{condition:<20} completos={item['complete']}/{item['rows']} "
            f"(diagnóstico={item['complete_diagnostic']}/{item['rows']}) "
            f"decisão={item['decision_ok']}/{item['decision_total']} "
            f"(fato={item['decision_fact_only']}) "
            f"motivo={item['reason_ok']}/{item['reason_total']} "
            f"concluído={item['completed_ok']}/{item['completed_total']} "
            f"pendência={item['pending_ok']}/{item['pending_total']} "
            f"refs_sem_falha={item['reference_support']}/{item['rows']} "
            f"condições={item['conditions_ok']}/{item['conditions_total']} "
            f"projeto={item['project_ok']}/{item['project_total']} "
            f"prompt={item['prompt_tokens']} completion={item['completion_tokens']}"
        )
    lines.append(
        f"tokens reais: prompt={result['real_tokens']['prompt_tokens']} "
        f"completion={result['real_tokens']['completion_tokens']}"
    )
    (out / "real_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "results": str(out / "real_results.json"),
        "review": str(out / "real_review_blind.json"),
        "summary": str(out / "real_summary.txt"),
    }
