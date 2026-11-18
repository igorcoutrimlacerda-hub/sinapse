"""Validação local dos pares preparados para a comparação real.

Não executa o DeepSeek. Reconstrói as duas condições congeladas, confere se
cada referência foi realmente enviada e se o trecho sustenta o elemento ao qual
está ligado. Dunas é checada quanto à condição e aos dois caminhos; Norte/Sul,
quanto à atribuição ao projeto solicitado.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .deepseek import DeepSeekSettings, load_deepseek_settings
from .models import Message, load_messages
from .resumption import (
    ELEMENTS,
    _memory_ids,
    _multipurpose_context,
    _reference_support,
    _run_agent,
    cases,
    evaluate_answer,
)
from .scale_benchmark import _id_in_messages
from .text import normalize_text


def _element_ids(case, name: str) -> tuple[str, ...]:
    return getattr(case, f"{name}_ids", ())


def _validate_condition(
    case,
    label: str,
    selected: Sequence[str],
    messages: Sequence[Mapping[str, str]],
    answer: str,
    corpus: Sequence[Message],
) -> dict[str, Any]:
    selected_set = set(selected)
    support, support_failures = _reference_support(case, corpus, selected)
    names = list(case.components) + (
        ["condition"] if case.condition_ids else []
    )
    missing = [
        name
        for name in names
        if _element_ids(case, name) and not (selected_set & set(_element_ids(case, name)))
    ]
    selected_refs = set()
    for name in list(ELEMENTS) + ["condition"]:
        selected_refs |= selected_set & set(_element_ids(case, name))
    not_sent = [
        message_id
        for message_id in selected_refs
        if not _id_in_messages(message_id, messages)
    ]
    checks = evaluate_answer(case, answer)[1]
    complete = all(checks[name] for name in case.components)
    conditions_ok = True
    paths: dict[str, bool] | None = None
    if case.condition_ids:
        normalized = normalize_text(answer)
        conditions_ok = not checks["conditions_omitted"]
        if case.id == "proximo_condicionado":
            paths = {
                "condition": (
                    "30" in normalized
                    and ("abaixo" in normalized or "se " in normalized)
                ),
                "export_path": "exportar" in normalized,
                "fallback_path": (
                    "oversampling" in normalized
                    or "reduz" in normalized
                    or "senao" in normalized
                ),
            }
    project_ok = True
    wrong_selected: list[str] = []
    wrong_terms: list[str] = []
    if case.wrong_project_ids:
        wrong_selected = [
            message_id
            for message_id in case.wrong_project_ids
            if message_id in selected_set
        ]
        normalized = normalize_text(answer)
        wrong_terms = [
            term
            for term in case.wrong_project_terms
            if normalize_text(term) in normalized
        ]
        project_ok = not wrong_selected and not wrong_terms
    integrity = not support_failures and not not_sent
    return {
        "condition": label,
        "selected": list(selected),
        "missing_elements": missing,
        "references_sent": not not_sent,
        "reference_failures": support_failures,
        "reference_support": support,
        "complete": complete,
        "integrity": integrity,
        "conditions_ok": conditions_ok,
        "conditions_omitted": checks["conditions_omitted"],
        "condition_paths": paths,
        "project_ok": project_ok,
        "wrong_project_selected": wrong_selected,
        "wrong_project_terms": wrong_terms,
        "answer": answer,
        "checks": checks,
    }


def validate_real_comparison(
    pairs_path: str | Path = "benchmark_out_resumption_round3/real_comparison.json",
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    dataset: str | Path = "benchmark/resumption_sessions.json",
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    data = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    corpus = load_messages(dataset)
    by_case = {case.id: case for case in cases()}
    memory_corpora = {
        case.id: [
            message for message in corpus if message.id not in case.recent_ids
        ]
        for case in cases()
    }
    results = []
    for pair in data["pairs"]:
        case = by_case[pair["case"]]
        single_turn = _run_agent(
            case, memory_corpora[case.id], config, settings
        )
        single_selected = _memory_ids(single_turn)
        if case.open_resumption:
            multi_messages, multi_selected, _, _ = _multipurpose_context(
                case, memory_corpora[case.id], config, settings
            )
        else:
            multi_messages = single_turn.messages
            multi_selected = single_selected
        single = _validate_condition(
            case,
            "sinapse",
            single_selected,
            single_turn.messages,
            pair["sinapse_answer"],
            corpus,
        )
        multi = _validate_condition(
            case,
            "sinapse_multipurpose",
            multi_selected,
            multi_messages,
            pair["multipurpose_answer"],
            corpus,
        )
        pair_valid = single["integrity"] and multi["integrity"]
        selected_match = (
            set(pair["sinapse_selected"]) == set(single_selected)
            and set(pair["multipurpose_selected"]) == set(multi_selected)
        )
        results.append(
            {
                "case": case.id,
                "question": pair["question"],
                "expected_ids": pair["expected_ids"],
                "pair_selection_matches_frozen": selected_match,
                "pair_valid": pair_valid,
                "single": single,
                "multipurpose": multi,
            }
        )
    dunas = [
        result
        for result in results
        if result["case"] == "proximo_condicionado"
    ]
    project = [
        result
        for result in results
        if result["case"].startswith("intercalado_")
    ]
    summary = {
        "pairs": len(results),
        "pair_valid": sum(result["pair_valid"] for result in results),
        "selection_matches_frozen": sum(
            result["pair_selection_matches_frozen"] for result in results
        ),
        "single_complete": sum(
            result["single"]["complete"] for result in results
        ),
        "multipurpose_complete": sum(
            result["multipurpose"]["complete"] for result in results
        ),
        "single_integrity": sum(
            result["single"]["integrity"] for result in results
        ),
        "multipurpose_integrity": sum(
            result["multipurpose"]["integrity"] for result in results
        ),
        "dunas_condition_ok": sum(
            result["multipurpose"]["conditions_ok"] for result in dunas
        ),
        "dunas_paths_ok": sum(
            bool(result["multipurpose"]["condition_paths"])
            and all(result["multipurpose"]["condition_paths"].values())
            for result in dunas
        ),
        "dunas_single_condition_ok": sum(
            result["single"]["conditions_ok"] for result in dunas
        ),
        "dunas_single_paths_ok": sum(
            bool(result["single"]["condition_paths"])
            and all(result["single"]["condition_paths"].values())
            for result in dunas
        ),
        "project_ok": sum(
            result["multipurpose"]["project_ok"] for result in project
        ),
        "project_single_ok": sum(
            result["single"]["project_ok"] for result in project
        ),
        "project_pairs": len(project),
        "project_total_ok": (
            sum(result["multipurpose"]["project_ok"] for result in project)
            + sum(result["single"]["project_ok"] for result in project)
        ),
        "project_total": 2 * len(project),
        "human_review_pending": True,
        "real_evaluation_available": False,
        "note": (
            "Validação local apenas; as respostas reais dependem de acesso ao "
            "deepseek-flash."
        ),
    }
    return {
        "pairs_file": str(pairs_path),
        "dataset": str(dataset),
        "review_files_preserved": True,
        "rubric_preserved": True,
        "summary": summary,
        "pairs": results,
    }


def export_validation(
    result: Mapping[str, Any], out_path: str | Path
) -> str:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)
