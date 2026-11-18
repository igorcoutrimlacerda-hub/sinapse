"""Rubrica v2 da retomada: paráfrases, ação/objeto/parâmetros e condições.

A v1 congelada permanece intacta. Esta versão é calibrada nos dez pares reais
revisados e em casos sintéticos novos, sem novas chamadas ao modelo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .resumption import cases
from .text import normalize_text

DECISION_CUES = (
    "decisao vigente",
    "esta valendo",
    "vale hoje",
    "preset atual",
    "agora e o",
    "e o preset",
    "uso o",
    "usa o",
    "estou usando",
    "troquei para",
    "trocou para",
    "trocar para",
    "troquei o",
    "trocou o",
    "trocar o",
    "trocado para",
    "trocado o",
    "continua sendo",
    "continua o",
    "adotado",
    "escolhi",
    "decidi",
)
REASON_CAUSES = ("cpu", "40", "alta", "caiu", "22", "estoura", "peso", "porque")
PENDING_ACTIONS = (
    "testar",
    "teste",
    "rodar",
    "validar",
    "medir",
    "verificar",
    "comecar",
    "começar",
    "iniciar",
    "exportar",
)
NO_PENDING_ACK = (
    "em aberto",
    "sem proximo passo",
    "nao defini",
    "nao definido",
    "nao ha proximo",
    "nenhum proximo",
    "ainda nao",
)


@dataclass(frozen=True)
class RubricSpec:
    id: str
    components: tuple[str, ...]
    decision_facts: tuple[str, ...] = ()
    old_current_terms: tuple[str, ...] = ()
    reason_facts: tuple[str, ...] = ()
    completed_facts: tuple[str, ...] = ()
    pending_action: tuple[str, ...] = PENDING_ACTIONS
    pending_object: tuple[str, ...] = ()
    pending_params: tuple[str, ...] = ()
    no_pending: bool = False
    condition_facts: tuple[str, ...] = ()
    condition_paths: tuple[tuple[str, ...], ...] = ()


PENDING_SPECS = {
    "retomada_geral": (("introducao",), ("96 khz",)),
    "proximo_teste": (("introducao",), ("96 khz",)),
    "pendencia_concluida": (("versao final",), ()),
    "pendencia_substituida": (("ponte",), ("48 khz",)),
    "pendencia_substituida_costa": (("ponte",), ("48 khz",)),
    "proximo_condicionado": (("ponte",), ("96 khz",)),
    "intercalado_norte": (("ponte",), ("48 khz",)),
    "intercalado_sul": (("refrao",), ()),
}


def specs() -> dict[str, RubricSpec]:
    result = {}
    for case in cases():
        objects, params = PENDING_SPECS.get(case.id, ((), ()))
        result[case.id] = RubricSpec(
            id=case.id,
            components=case.components,
            decision_facts=case.decision_facts,
            old_current_terms=case.wrong_current_terms,
            reason_facts=case.reason_facts,
            completed_facts=case.completed_facts,
            pending_object=objects,
            pending_params=params,
            no_pending=case.no_pending,
            condition_facts=case.condition_facts,
            condition_paths=(
                (("exportar",), ("oversampling", "reduz"))
                if case.id == "proximo_condicionado"
                else ()
            ),
        )
    return result


def _occurrences(text: str, needle: str) -> list[int]:
    positions = []
    start = text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)
    return positions


def _near(text: str, objects: Sequence[str], actions: Sequence[str], window: int = 70) -> bool:
    for obj in objects:
        for obj_pos in _occurrences(text, obj):
            for action in actions:
                for action_pos in _occurrences(text, action):
                    if abs(obj_pos - action_pos) <= window:
                        return True
    return False


def _old_as_current(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        for position in _occurrences(text, normalize_text(term)):
            prefix = text[max(0, position - 12) : position]
            if not any(neg in prefix for neg in ("nao ", "nunca ", "deixei de ")):
                return True
    return False


def evaluate_v2(
    spec: RubricSpec, answer: str
) -> tuple[float, dict[str, Any]]:
    text = normalize_text(answer)
    current_hit = any(
        normalize_text(fact) in text for fact in spec.decision_facts
    )
    current_cue = any(cue in text for cue in DECISION_CUES)
    old_current = _old_as_current(text, spec.old_current_terms)
    decision = current_hit and current_cue and not old_current

    reason = any(
        normalize_text(fact) in text for fact in spec.reason_facts
    ) and any(cause in text for cause in REASON_CAUSES)

    completed = bool(spec.completed_facts) and all(
        normalize_text(fact) in text for fact in spec.completed_facts
    )

    if spec.no_pending:
        pending = any(ack in text for ack in NO_PENDING_ACK)
    else:
        action_near_object = _near(
            text, spec.pending_object, spec.pending_action
        )
        params_ok = all(
            normalize_text(param) in text for param in spec.pending_params
        )
        pending = action_near_object and params_ok

    condition = True
    if spec.condition_facts:
        condition = all(
            normalize_text(fact) in text for fact in spec.condition_facts
        ) and all(
            any(term in text for term in group)
            for group in spec.condition_paths
        )

    checks = {
        "decision": decision,
        "reason": reason,
        "completed": completed,
        "pending": pending,
        "condition": condition,
        "old_or_ambiguous": old_current,
    }
    components = [checks[name] for name in spec.components]
    score = sum(components) / len(components) if components else 1.0
    complete = all(components) and condition
    return round(score, 4), {
        **checks,
        "components": list(spec.components),
        "complete": complete,
    }


def load_synthetic_cases(
    path: str | Path = "benchmark/resumption_rubric_v2_cases.json",
) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _spec_from_dict(raw: Mapping[str, Any]) -> RubricSpec:
    return RubricSpec(
        id=str(raw["id"]),
        components=tuple(raw["components"]),
        decision_facts=tuple(raw.get("decision_facts", ())),
        old_current_terms=tuple(raw.get("old_current_terms", ())),
        reason_facts=tuple(raw.get("reason_facts", ())),
        completed_facts=tuple(raw.get("completed_facts", ())),
        pending_action=tuple(raw.get("pending_action", PENDING_ACTIONS)),
        pending_object=tuple(raw.get("pending_object", ())),
        pending_params=tuple(raw.get("pending_params", ())),
        no_pending=bool(raw.get("no_pending", False)),
        condition_facts=tuple(raw.get("condition_facts", ())),
        condition_paths=tuple(
            tuple(group) for group in raw.get("condition_paths", ())
        ),
    )


def _human_rows(
    review_path: str | Path,
    results_path: str | Path,
    consolidated_path: str | Path,
) -> list[dict[str, Any]]:
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    consolidated = json.loads(
        Path(consolidated_path).read_text(encoding="utf-8")
    )
    mapping = {
        item["review_id"]: item["conditions"]
        for item in consolidated["revealed_conditions"]
    }
    rows = []
    for pair in review:
        conditions = mapping[pair["review_id"]]
        for label in ("A", "B"):
            condition = conditions[label]
            row = next(
                row
                for row in results["rows"]
                if row["case"] == pair["case"]
                and row["condition"] == condition
                and row["answer"] == pair["answers"][label]
            )
            rows.append(
                {
                    "review_id": pair["review_id"],
                    "case": pair["case"],
                    "condition": condition,
                    "answer": pair["answers"][label],
                    "human": pair["human"][label],
                    "automatic": row["checks"],
                }
            )
    return rows


def calibrate(
    review_path: str | Path = "benchmark_out_resumption_real/real_review_blind.json",
    results_path: str | Path = "benchmark_out_resumption_real/real_results.json",
    consolidated_path: str | Path = "benchmark_out_resumption_real/human_consolidated.json",
    synthetic_path: str | Path = "benchmark/resumption_rubric_v2_cases.json",
) -> dict[str, Any]:
    spec_by_case = specs()
    rows = _human_rows(review_path, results_path, consolidated_path)
    criteria = ("decision", "reason", "completed", "pending", "condition")
    counts = {
        name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for name in criteria
    }
    detailed = []
    for row in rows:
        spec = spec_by_case[row["case"]]
        _, checks = evaluate_v2(spec, row["answer"])
        human = row["human"]
        applicable = [
            name for name in criteria if name in human
        ]
        for name in applicable:
            h = human[name]["score"]
            v = int(checks[name])
            if h == 1 and v == 1:
                counts[name]["tp"] += 1
            elif h == 0 and v == 0:
                counts[name]["tn"] += 1
            elif h == 0 and v == 1:
                counts[name]["fp"] += 1
            elif h == 1 and v == 0:
                counts[name]["fn"] += 1
        detailed.append(
            {
                "review_id": row["review_id"],
                "case": row["case"],
                "condition": row["condition"],
                "human": {
                    name: human[name]["score"] for name in applicable
                },
                "v2": {name: checks[name] for name in applicable},
            }
        )
    synthetic = []
    for case in load_synthetic_cases(synthetic_path):
        spec = _spec_from_dict(case)
        _, checks = evaluate_v2(spec, case["answer"])
        synthetic.append(
            {
                "id": case["id"],
                "kind": case.get("kind", ""),
                "gold": case["gold"],
                "v2": {name: checks[name] for name in case["gold"]},
                "answer": case["answer"],
            }
        )
    return {
        "rows": len(rows),
        "criteria": counts,
        "details": detailed,
        "synthetic": synthetic,
    }


def export_calibration(
    result: Mapping[str, Any],
    out_path: str | Path = "benchmark_out_resumption_real/rubric_v2_calibration.json",
) -> str:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)
