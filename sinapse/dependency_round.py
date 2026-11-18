"""Dependências condicionais: variante conservadora vs referência por propósito.

Só omite uma dependência presente quando o turno atual ou o histórico
identificam, de forma inequívoca, a mesma entidade e decisão. Ambiguidade,
duas entidades ou duas decisões preservam o vínculo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter
from .config import load_config
from .context_reduction import _initial_state, _run_action, _train_more_model
from .curiosity_learning import _id_in_messages
from .deepseek import ChatCompletionResponse, ChatTransport, DeepSeekSettings, build_request_body, load_deepseek_settings
from .genetic_policy import REFERENCE, _choose_action
from .purpose_answer import SimulatedPurposeClient, _checks, _render_selection
from .purpose_selection import Purpose, PurposeCase, _cases as purpose_cases, _select_purpose, _train, _load, _recent
from .removal_round import _render_exact
from .text import normalize_text, tokenize

BASE = "data/agent_conversation.json"


def _cases() -> list[PurposeCase]:
    bike = _load(BASE, "benchmark/price_bicycle.json")
    bike_decisions = _load(BASE, "benchmark/bike_decisions.json")
    return [
        PurposeCase(
            "dep_bike_named",
            "bicicleta",
            _load(BASE),
            "c2",
            _recent("Falamos de bicicleta."),
            "Por que comprei a bicicleta?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), ("p1",), ("trilha",)),),
            "p1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "dep_short_history",
            "historico",
            _load(BASE),
            "c2",
            (
                RecentTurn("Comprei uma bicicleta para ir ao trabalho.", id="p1"),
                RecentTurn("Porque a trilha é mais curta.", id="p2"),
            ),
            "E por quê?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), (), ("trilha",)),),
            "p1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "dep_two_entities",
            "ambiguidade",
            _load(BASE),
            "c1",
            _recent("Falamos do Serum 2 e do Piano V."),
            "Por que escolhi ele?",
            (Purpose("m", "motivo", "serum", "por_que", ("m2",), ("m1",), ("licenca",)),),
            "m1",
            memory_limit=120,
            split="eval",
        ),
        PurposeCase(
            "dep_two_decisions",
            "duas_decisoes",
            bike_decisions,
            "c2",
            _recent("Falamos de bicicleta."),
            "Por que comprei a bicicleta?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("p2",), ("p1",), ("trilha",)),),
            "p1",
            memory_limit=120,
            split="eval",
        ),
        PurposeCase(
            "dep_two_purposes",
            "misto",
            bike,
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
            "dep_redundant",
            "redundancia",
            bike,
            "c2",
            _recent("Falamos de bicicleta."),
            "Quanto custou a bicicleta?",
            (Purpose("p", "preco", "bicicleta", "o_que", ("p3",), (), ("1200",)),),
            "p1",
            memory_limit=140,
            split="eval",
        ),
    ]


def _decision_terms(case_id: str) -> tuple[str, ...]:
    return {
        "dep_bike_named": ("comprei", "comprou", "compra", "comprar"),
        "dep_short_history": ("comprei", "comprou", "compra", "comprar"),
        "dep_two_entities": ("escolhi", "escolheu", "escolha"),
        "dep_two_decisions": ("comprei", "comprou", "compra", "comprar"),
        "dep_two_purposes": ("comprei", "comprou", "compra", "comprar"),
        "dep_redundant": ("custou", "custa", "preco", "valor"),
    }.get(case_id, ())


def _can_omit(case: PurposeCase, state, selected: Sequence[str]) -> bool:
    text = normalize_text(case.question)
    question_tokens = tokenize(case.question)
    themes = {purpose.theme for purpose in case.purposes}
    named = any(tokenize(theme) & question_tokens for theme in themes)
    pronoun = any(
        token in normalize_text(case.question).split()
        for token in ("ele", "ela", "isso", "isto", "aquilo")
    )
    recent_entities = set()
    for turn in case.recent:
        for theme in themes:
            if tokenize(theme) & tokenize(turn.text):
                recent_entities.add(theme)
    decisions = [
        message
        for message in case.messages
        if any(
            term in normalize_text(message.text)
            for term in ("comprei", "troquei", "deixei", "escolhi", "agora")
        )
    ]
    same_theme_decisions = [
        message
        for message in decisions
        if any(tokenize(theme) & tokenize(message.text) for theme in themes)
    ]
    if pronoun and len(recent_entities) >= 2:
        return False
    if len(same_theme_decisions) >= 2:
        return False
    # entidade nomeada no turno ou histórico com item + decisão
    history_identifies = any(
        _id_in_messages(message.id, state.base_messages)
        and any(tokenize(theme) & tokenize(message.text) for theme in themes)
        for message in case.messages
    )
    return bool(named or history_identifies)


def _answer_checks(case: PurposeCase, answer: str) -> dict[str, Any]:
    base = _checks(case, answer)
    normalized = normalize_text(answer)
    decision_terms = _decision_terms(case.id)
    decision_ok = any(normalize_text(term) in normalized for term in decision_terms)
    ambiguity_ok = True
    if case.id == "dep_two_entities":
        ambiguity_ok = any(
            marker in normalized
            for marker in ("serum", "piano", "qual", "voce quer dizer", "esclarec")
        )
    purposes = base["purposes"]
    target = next(iter(purposes), None)
    preserve = [purpose_id for purpose_id in purposes if purpose_id != target]
    preserve_ok = all(purposes[purpose_id]["ok"] for purpose_id in preserve)
    fact_ok = all(
        (value.get("term_ok", True) or value.get("unavailable", False))
        for value in purposes.values()
    )
    entity_ok = all(
        (value.get("item_ok", True) or value.get("unavailable", False))
        for value in purposes.values()
    )
    components = [
        1.0 if fact_ok else 0.0,
        1.0 if entity_ok else 0.0,
        1.0 if decision_ok else 0.0,
        1.0 if ambiguity_ok else 0.0,
        1.0 if preserve_ok else 0.0,
    ]
    return {
        **base,
        "fact_ok": fact_ok,
        "entity_ok": entity_ok,
        "decision_ok": decision_ok,
        "ambiguity_ok": ambiguity_ok,
        "preserve_ok": preserve_ok,
        "score": round(sum(components) / len(components), 4),
    }


def _verdict(reference_checks, condition_checks, case: PurposeCase, omitted: bool) -> str:
    if not omitted:
        return "sem_efeito_na_requisicao"
    target = next(iter(reference_checks["purposes"]), None)
    if target is None:
        return "inconclusiva"
    base = reference_checks["purposes"][target]["ok"]
    after = condition_checks["purposes"].get(target, {}).get("ok", False)
    preserved = condition_checks.get("preserve_ok", True)
    if not base:
        return "inconclusiva_base"
    if after and preserved:
        return "preservou"
    if not after and preserved:
        return "prejudicou"
    return "prejudicou_com_perda_colateral"


def run_dependency_round(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    train_cases, _ = purpose_cases()
    models = _train(train_cases, config)
    more = _train_more_model(config)
    cases = _cases()
    executed = 0
    entries = []
    for case in cases:
        state = _initial_state(case, config)
        action = _choose_action(more, state, REFERENCE)
        current = _run_action(action, state, config)
        purpose = _select_purpose(case, models, state, current.get("candidate_ids", []))
        reference_selection = _render_exact(state, purpose["selected"], ())
        planned = case.seed_id  # dependência semente planejada
        can_omit = planned in purpose["selected"] and _can_omit(case, state, purpose["selected"])
        conservative_selection = _render_exact(
            state, purpose["selected"], (planned,) if can_omit else ()
        )
        forced_selection = _render_exact(
            state, purpose["selected"], (planned,) if planned in purpose["selected"] else ()
        )
        conditions = {
            "reference": reference_selection,
            "conservative": conservative_selection,
            "forced": forced_selection,
        }
        cache: dict[str, dict[str, Any]] = {}
        results = {}
        for name, selection in conditions.items():
            key = json.dumps(selection["messages"], ensure_ascii=False, sort_keys=True)
            if key in cache:
                results[name] = dict(cache[key])
                results[name]["reused"] = True
                continue
            request = build_request_body(selection["messages"], settings)
            answer_client = (
                SimulatedPurposeClient(case, selection["messages"])
                if client is None
                else client
            )
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
                "selected": selection["selected"],
                "removed": [] if name == "reference" else (
                    [planned] if planned in purpose["selected"] and planned not in selection["selected"] else []
                ),
                "messages": [dict(message) for message in selection["messages"]],
                "answer": answer,
                "checks": _answer_checks(case, answer),
                "memory_tokens": selection["memory_tokens"],
                "token_source": "api" if usage.get("prompt_tokens") is not None else "estimativa_local",
                "usage": usage,
                "error": error,
                "reused": False,
            }
            cache[key] = result
            results[name] = result
            executed += 1
        conservative_omitted = (
            planned in purpose["selected"] and planned not in conservative_selection["selected"]
        )
        forced_omitted = planned in purpose["selected"] and planned not in forced_selection["selected"]
        entries.append(
            {
                "case": case.id,
                "group": "regression" if case.id == "dep_bike_named" else "new",
                "planned_dependency": planned,
                "reference_ids": reference_selection["selected"],
                "conservative_ids": conservative_selection["selected"],
                "forced_ids": forced_selection["selected"],
                "can_omit": can_omit,
                "local_only_dependency_change": (
                    set(reference_selection["selected"]) ^ set(conservative_selection["selected"])
                    <= {planned}
                ),
                "conditions": results,
                "conservative_omitted": conservative_omitted,
                "forced_omitted": forced_omitted,
                "verdict": _verdict(
                    results["reference"]["checks"],
                    results["conservative"]["checks"],
                    case,
                    conservative_omitted,
                ),
            }
        )
    return {
        "cases": len(entries),
        "calls": executed,
        "entries": entries,
        "comparisons": {
            "conservative_preserved": sum(1 for e in entries if e["verdict"] == "preservou"),
            "conservative_harmed": sum(1 for e in entries if e["verdict"].startswith("prejudicou")),
            "no_effect": sum(1 for e in entries if e["verdict"] == "sem_efeito_na_requisicao"),
        },
        "human_review_pending": True,
        "previous_17_pending_preserved": True,
    }


def export_dependency_review(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dependency_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, entry in enumerate(result["entries"], start=1):
        names = list(entry["conditions"])
        labels = {
            chr(ord("A") + position): entry["conditions"][name]["answer"]
            for position, name in enumerate(names)
        }
        review.append(
            {
                "review_id": f"E{index:02d}",
                "case": entry["case"],
                "group": entry["group"],
                "answers": labels,
                "rubric": "Fato, atribuição à entidade, decisão correta, ambiguidade e preservação das demais partes.",
                "human_review": {
                    "reviewed": False,
                    "approved_answers": [],
                    "divergences": [],
                    "notes": "",
                },
            }
        )
    (out / "dependency_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "dependency_results.json"),
        "review": str(out / "dependency_review.json"),
    }
