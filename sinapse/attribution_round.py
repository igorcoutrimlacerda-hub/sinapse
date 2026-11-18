"""Avaliação de atribuição: entidade/decisão, dependência e ambiguidade."""

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
ATTRIBUTION = _load("benchmark/attribution_c3.json")


def _cases() -> list[PurposeCase]:
    return [
        PurposeCase(
            "attr_named_entity",
            "entidade_nomeada",
            ATTRIBUTION,
            "c3",
            _recent("Falamos da Aurora e da Nebula."),
            "Por que escolhi a Aurora?",
            (Purpose("m", "motivo", "aurora", "por_que", ("z2",), ("z1",), ("licenca",)),),
            "z1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "attr_ambiguous_entity",
            "duas_entidades",
            ATTRIBUTION,
            "c3",
            _recent("Falamos da Aurora e da Nebula."),
            "E por que escolhi?",
            (Purpose("m", "motivo", "aurora", "por_que", ("z2",), ("z1",), ("licenca",)),),
            "z1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "attr_named_decision",
            "decisao_nomeada",
            ATTRIBUTION,
            "c3",
            _recent("Falamos da compra e da troca da bicicleta."),
            "Por que comprei a bicicleta?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("x2",), ("x1",), ("trilha",)),),
            "x1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "attr_ambiguous_decision",
            "duas_decisoes",
            ATTRIBUTION,
            "c3",
            _recent("Falamos da compra e da troca da bicicleta."),
            "E por quê?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("x2",), ("x1",), ("trilha",)),),
            "x1",
            memory_limit=100,
            split="eval",
        ),
        PurposeCase(
            "attr_named_trade",
            "decisao_nomeada",
            ATTRIBUTION,
            "c3",
            _recent("Falamos da compra e da troca da bicicleta."),
            "Por que troquei a bicicleta?",
            (Purpose("m", "motivo", "bicicleta", "por_que", ("w2",), ("w1",), ("trilha",)),),
            "w1",
            memory_limit=100,
            split="eval",
        ),
    ]


def _decision_terms(case_id: str) -> tuple[str, ...]:
    return {
        "attr_named_entity": ("escolhi", "escolheu", "escolha"),
        "attr_ambiguous_entity": ("escolhi", "escolheu", "escolha"),
        "attr_named_decision": ("comprei", "comprou", "compra"),
        "attr_ambiguous_decision": ("comprei", "comprou", "compra"),
        "attr_named_trade": ("troquei", "trocou", "troca"),
    }.get(case_id, ())


def _can_omit(case: PurposeCase, state) -> bool:
    tokens = tokenize(case.question)
    normalized = normalize_text(case.question)
    theme_tokens = tokenize(case.purposes[0].theme)
    entity_named = bool(tokens & theme_tokens)
    decision_terms = _decision_terms(case.id)
    decision_named = any(term in normalized for term in decision_terms)
    pronoun = any(
        word in normalized.split() for word in ("ele", "ela", "isso", "isto", "aquilo")
    )
    if pronoun and not entity_named:
        return False
    recent_entities = {
        purpose.theme
        for purpose in case.purposes
        for turn in case.recent
        if tokenize(purpose.theme) & tokenize(turn.text)
    }
    if len(recent_entities) >= 2 and not entity_named:
        return False
    entity_vocabulary = set()
    for message in case.messages:
        for value in message.values("o_que"):
            entity_vocabulary.update(tokenize(value))
    recent_entity_hits = set()
    for turn in case.recent:
        recent_entity_hits.update(tokenize(turn.text) & entity_vocabulary)
    if len(recent_entity_hits) >= 2 and not entity_named:
        return False
    if entity_named:
        return True
    if decision_named and not pronoun:
        return True
    return False


def _answer_checks(case: PurposeCase, answer: str, parent_present: bool) -> dict[str, Any]:
    base = _checks(case, answer)
    normalized = normalize_text(answer)
    purposes = base["purposes"]
    target = next(iter(purposes), None)
    fact_ok = bool(purposes[target]["term_ok"]) if target else False
    entity_ok = bool(purposes[target]["item_ok"]) if target else False
    decision_ok = any(
        normalize_text(term) in normalized for term in _decision_terms(case.id)
    )
    ambiguous = case.id in {"attr_ambiguous_entity", "attr_ambiguous_decision"}
    if not ambiguous:
        ambiguity_ok = True
    elif parent_present:
        ambiguity_ok = True
    else:
        clarification = any(
            marker in normalized
            for marker in ("qual", "voce quer dizer", "quer dizer", "aurora ou nebula", "compra ou troca")
        )
        conditional = any(
            phrase in normalized
            for phrase in ("se for", "se a pergunta", "caso seja", "pode ser")
        )
        ambiguity_ok = clarification or conditional
        if not ambiguity_ok and entity_ok:
            # atribuição categórica sem suporte no contexto ambíguo
            ambiguity_ok = False
    components = [
        1.0 if fact_ok else 0.0,
        1.0 if entity_ok else 0.0,
        1.0 if decision_ok else 0.0,
        1.0 if ambiguity_ok else 0.0,
    ]
    return {
        **base,
        "fact_ok": fact_ok,
        "entity_ok": entity_ok,
        "decision_ok": decision_ok,
        "ambiguity_ok": ambiguity_ok,
        "score": round(sum(components) / len(components), 4),
    }


def _verdict(base, condition, omitted: bool, case: PurposeCase, parent_present: bool) -> str:
    if not omitted:
        return "sem_efeito_na_requisicao"
    if case.id.startswith("attr_ambiguous") and not parent_present:
        return "ambiguidade_exposta"
    if condition["checks"]["score"] >= base["checks"]["score"]:
        return "preservou"
    return "prejudicou"


def run_attribution_round(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    train_cases, _ = purpose_cases()
    models = _train(train_cases, config)
    more = _train_more_model(config)
    entries = []
    calls = 0
    for case in _cases():
        state = _initial_state(case, config)
        action = _choose_action(more, state, REFERENCE)
        current = _run_action(action, state, config)
        purpose = _select_purpose(case, models, state, current.get("candidate_ids", []))
        parent = case.seed_id
        reference = _render_exact(state, purpose["selected"], ())
        can_omit = parent in purpose["selected"] and _can_omit(case, state)
        conservative = _render_exact(
            state, purpose["selected"], (parent,) if can_omit else ()
        )
        forced = _render_exact(
            state, purpose["selected"], (parent,) if parent in purpose["selected"] else ()
        )
        conditions = {"reference": reference, "conservative": conservative, "forced": forced}
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
                    [parent] if parent in purpose["selected"] and parent not in selection["selected"] else []
                ),
                "messages": [dict(message) for message in selection["messages"]],
                "answer": answer,
                "checks": _answer_checks(
                    case,
                    answer,
                    parent_present=_id_in_messages(parent, selection["messages"]),
                ),
                "memory_tokens": selection["memory_tokens"],
                "token_source": "api" if usage.get("prompt_tokens") is not None else "estimativa_local",
                "usage": usage,
                "error": error,
                "reused": False,
            }
            cache[key] = result
            results[name] = result
            calls += 1
        conservative_omitted = parent in reference["selected"] and parent not in conservative["selected"]
        forced_omitted = parent in reference["selected"] and parent not in forced["selected"]
        entries.append(
            {
                "case": case.id,
                "kind": case.family,
                "parent": parent,
                "reference_ids": reference["selected"],
                "conservative_ids": conservative["selected"],
                "forced_ids": forced["selected"],
                "can_omit": can_omit,
                "local_only_parent_change": (
                    set(reference["selected"]) ^ set(conservative["selected"]) <= {parent}
                    and set(reference["selected"]) ^ set(forced["selected"]) <= {parent}
                ),
                "no_equivalent_clue": (
                    parent not in forced["selected"]
                    and (
                        next(
                            (message for message in case.messages if message.id == parent),
                            None,
                        )
                        is None
                        or normalize_text(
                            next(
                                (message for message in case.messages if message.id == parent),
                                None,
                            ).text
                        )
                        not in normalize_text(
                            "\n".join(str(message["content"]) for message in forced["messages"])
                        )
                    )
                ),
                "conditions": results,
                "conservative_verdict": _verdict(
                    results["reference"], results["conservative"], conservative_omitted, case, True
                ),
                "forced_verdict": _verdict(
                    results["reference"],
                    results["forced"],
                    forced_omitted,
                    case,
                    _id_in_messages(parent, forced["messages"]),
                ),
            }
        )
    return {
        "cases": len(entries),
        "calls": calls,
        "entries": entries,
        "human_review_pending": True,
        "previous_23_pending_preserved": True,
    }


def export_attribution_review(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, entry in enumerate(result["entries"], start=1):
        labels = {
            chr(ord("A") + position): entry["conditions"][name]["answer"]
            for position, name in enumerate(entry["conditions"])
        }
        review.append(
            {
                "review_id": f"F{index:02d}",
                "case": entry["case"],
                "answers": labels,
                "rubric": "Fato, entidade, decisão, ambiguidade; atribuição categórica sem suporte é erro.",
                "human_review": {
                    "reviewed": False,
                    "approved_answers": [],
                    "divergences": [],
                    "notes": "",
                },
            }
        )
    (out / "attribution_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "attribution_results.json"),
        "review": str(out / "attribution_review.json"),
    }
