"""Revisão humana cega dos pares reais e consolidação posterior.

Prepara o formulário A/B sem revelar condições, mantém a rubrica automática e
o diagnóstico em registros separados, e só revela o mapa A/B depois que todas
as notas humanas aplicáveis forem preenchidas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import load_messages
from .resumption import cases

DATASET = "benchmark/resumption_sessions.json"
ELEMENTS = ("decision", "reason", "completed", "pending", "condition")
DECISION_GUIDE = (
    "Aceite paráfrases. Marque current_identified quando a resposta diz o que "
    "vale agora; reference_supports quando a referência citada sustenta essa "
    "leitura; old_or_ambiguous como true se a resposta tratar uma alternativa "
    "antiga ou ambígua como decisão vigente."
)


def _applicable(case) -> list[str]:
    names = list(case.components)
    if case.condition_ids:
        names.append("condition")
    return names


def _blank_human(names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if name == "decision":
            result[name] = {
                "applicable": True,
                "score": None,
                "current_identified": None,
                "reference_supports": None,
                "old_or_ambiguous": None,
                "notes": "",
            }
        else:
            result[name] = {
                "applicable": True,
                "score": None,
                "notes": "",
            }
    return result


def _sources(answers: Mapping[str, str], corpus: Sequence[Any]) -> list[dict[str, str]]:
    known = {message.id: message for message in corpus}
    pattern = re.compile(r"\b([a-z]{1,3}\d{1,3})\b")
    cited: set[str] = set()
    for answer in answers.values():
        cited.update(pattern.findall(answer))
    return [
        {
            "id": message_id,
            "conversation_id": known[message_id].conversation_id,
            "text": known[message_id].text,
        }
        for message_id in sorted(cited)
        if message_id in known
    ]


def prepare_blind_review(
    review_path: str | Path,
    real_results_path: str | Path,
    out_path: str | Path | None = None,
    dataset: str | Path = DATASET,
) -> str:
    """Acrescenta o formulário humano ao arquivo cego, sem tocar nas respostas."""

    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    results = json.loads(Path(real_results_path).read_text(encoding="utf-8"))
    case_by_id = {case.id: case for case in cases()}
    corpus = load_messages(dataset)
    # Confere que as respostas existentes são exatamente as dos resultados reais.
    real_answers = {
        (row["case"], row["condition"]): row["answer"]
        for row in results["rows"]
    }
    valid_answers = {value for value in real_answers.values()}
    prepared = []
    for pair in review:
        case = case_by_id[pair["case"]]
        for label in ("A", "B"):
            if pair["answers"][label] not in valid_answers:
                raise ValueError(
                    f"{pair['review_id']}: resposta {label} não está nos resultados reais"
                )
        names = _applicable(case)
        prepared.append(
            {
                "schema_version": 1,
                "review_id": pair["review_id"],
                "case": pair["case"],
                "question": pair["question"],
                "rubric": pair["rubric"],
                "decision_guide": DECISION_GUIDE,
                "applicable_elements": names,
                "answers": {
                    "A": pair["answers"]["A"],
                    "B": pair["answers"]["B"],
                },
                "sources": _sources(pair["answers"], corpus),
                "human": {
                    "A": _blank_human(names),
                    "B": _blank_human(names),
                },
                "human_review": {
                    "reviewed": False,
                    "reviewer": "",
                    "preferred": None,
                    "notes": "",
                },
            }
        )
    target = Path(out_path or review_path)
    target.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)


def _missing_fields(pair: Mapping[str, Any]) -> list[str]:
    missing = []
    for label in ("A", "B"):
        human = pair["human"][label]
        for name in pair["applicable_elements"]:
            field = human.get(name, {})
            if field.get("score") not in (0, 1):
                missing.append(f"{pair['review_id']}:{label}:{name}:score")
            if name == "decision":
                for sub in (
                    "current_identified",
                    "reference_supports",
                    "old_or_ambiguous",
                ):
                    if not isinstance(field.get(sub), bool):
                        missing.append(
                            f"{pair['review_id']}:{label}:decision:{sub}"
                        )
    if not pair.get("human_review", {}).get("reviewed"):
        missing.append(f"{pair['review_id']}:human_review.reviewed")
    if pair.get("human_review", {}).get("preferred") not in (
        "A",
        "B",
        "empate",
    ):
        missing.append(f"{pair['review_id']}:human_review.preferred")
    return missing


def _map_conditions(
    pair: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    mapping = {}
    for label in ("A", "B"):
        answer = pair["answers"][label]
        matches = [
            row
            for row in rows
            if row["case"] == pair["case"] and row["answer"] == answer
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{pair['review_id']}: resposta {label} com "
                f"{len(matches)} correspondências nos resultados reais"
            )
        mapping[label] = matches[0]["condition"]
    if len(set(mapping.values())) != 2:
        raise ValueError(
            f"{pair['review_id']}: A e B apontam para a mesma condição"
        )
    return mapping


def consolidate_review(
    review_path: str | Path,
    real_results_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    results = json.loads(Path(real_results_path).read_text(encoding="utf-8"))
    missing = [
        item for pair in review for item in _missing_fields(pair)
    ]
    if missing:
        return {
            "complete": False,
            "revealed": False,
            "missing": missing,
            "note": "As condições permanecem ocultas até todas as notas serem registradas.",
        }
    rows = results["rows"]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    divergences = []
    revealed = []
    preferences = {
        "sinapse": {"wins": 0, "ties": 0},
        "sinapse_multipurpose": {"wins": 0, "ties": 0},
    }
    for pair in review:
        mapping = _map_conditions(pair, rows)
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
        for label in ("A", "B"):
            condition = mapping[label]
            row = next(
                row
                for row in rows
                if row["case"] == pair["case"]
                and row["condition"] == condition
                and row["answer"] == pair["answers"][label]
            )
            by_condition.setdefault(condition, []).append(
                {
                    "review_id": pair["review_id"],
                    "case": pair["case"],
                    "human": pair["human"][label],
                    "automatic": row["checks"],
                    "automatic_complete": row["complete"],
                    "diagnostic_decision": row["decision_fact_present"],
                    "diagnostic_complete": row["complete_diagnostic"],
                }
            )
            for name in pair["applicable_elements"]:
                human_score = pair["human"][label][name]["score"]
                if name == "condition":
                    automatic = bool(row.get("conditions_ok"))
                else:
                    automatic = bool(row["checks"].get(name))
                if human_score != int(automatic):
                    divergences.append(
                        {
                            "review_id": pair["review_id"],
                            "case": pair["case"],
                            "condition": condition,
                            "element": name,
                            "human": human_score,
                            "automatic": int(automatic),
                        }
                    )
                if name == "decision" and human_score != int(
                    row["decision_fact_present"]
                ):
                    divergences.append(
                        {
                            "review_id": pair["review_id"],
                            "case": pair["case"],
                            "condition": condition,
                            "element": "decision_diagnostic",
                            "human": human_score,
                            "diagnostic": int(row["decision_fact_present"]),
                        }
                    )
    summary = {}
    for condition, items in by_condition.items():
        summary[condition] = {
            "rows": len(items),
            "elements": {
                name: {
                    "hits": sum(
                        item["human"][name]["score"] for item in items
                        if name in item["human"]
                    ),
                    "total": sum(
                        1 for item in items if name in item["human"]
                    ),
                }
                for name in ELEMENTS
            },
            "complete": sum(
                all(
                    item["human"][name]["score"] == 1
                    for name in item["human"]
                )
                for item in items
            ),
            "automatic_complete": sum(
                int(item["automatic_complete"]) for item in items
            ),
            "diagnostic_complete": sum(
                int(item["diagnostic_complete"]) for item in items
            ),
        }
    result = {
        "complete": True,
        "revealed": True,
        "reviewed_pairs": len(review),
        "revealed_conditions": revealed,
        "human_summary": summary,
        "preferences": preferences,
        "divergences": divergences,
        "review_method": (
            "revisão cega assistida por IA; respostas A/B sem rótulo de "
            "condição antes da consolidação"
        ),
        "automatic_preserved": str(real_results_path),
        "note": "Resultados humanos; rubrica automática e diagnóstico preservados separadamente.",
    }
    target = Path(out_path or "benchmark_out_resumption_real/human_consolidated.json")
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
