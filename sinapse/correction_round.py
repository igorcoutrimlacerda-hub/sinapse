"""Rodada da fonte de correção: referência `extra_context` vs fonte ativa.

Congela o seletor por propósito, `candidate_k=200`, a expansão por vínculo e o
orçamento de 160 tokens. A única mudança é a fonte que injeta revisões
(entidade + decisão) antes do corte. Os oito casos conhecidos rodam no corpus
original de 100k; as famílias novas rodam em um corpus de 100k com dois alvos
extras (t9/t10), fixados antes da comparação.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .config import load_config
from .models import Message
from .scale_benchmark import (
    QUERIES,
    ScaleQuery,
    _base_tokens,
    _id_in_messages,
    _percentile,
    _recent_for,
    _tokens,
    build_shortlist,
    generate_messages,
    target_messages,
)
from .scale_recovery import _coverage
from .whiteboard import MemoryAgent

KNOWN_QUERIES = tuple(query for query in QUERIES if query.expected_ids)

# Gabarito fixado antes da comparação. A pergunta sobre a decisão antiga
# mantém a correção no histórico, mas pede o motivo original (t2).
NEW_FAMILIES = (
    ScaleQuery(
        "correcao_pack",
        "Troquei qual pack da Aurora?",
        ("Falamos da Aurora.",),
        ("t9",),
        (),
    ),
    ScaleQuery(
        "correcao_preco",
        "Quanto paguei na Aurora depois da correção?",
        ("Falamos da Aurora.",),
        ("t10",),
        (),
    ),
    ScaleQuery(
        "decisao_antiga",
        "Por que eu escolhi a Aurora?",
        ("Falamos da correção da Aurora.", "Falamos da Aurora."),
        ("t2",),
        (),
    ),
    ScaleQuery(
        "resolvida_historico",
        "Por que eu escolhi a Aurora?",
        ("Porque eu já tinha comprado a licença da Aurora.", "Falamos da Aurora."),
        ("t2",),
        (),
        recent_ids=("t2", None),
    ),
)

GOLD_ROLES = {
    "correcao_pack": "correção sem reply_to, outra decisão (pack)",
    "correcao_preco": "correção sobre outra decisão (preço)",
    "decisao_antiga": "pergunta sobre a decisão antiga",
    "resolvida_historico": "consulta já resolvida pelo histórico",
}


def _index(messages: Sequence[Message]):
    token_sets = {index: _tokens(message) for index, message in enumerate(messages)}
    postings: dict[str, list[int]] = defaultdict(list)
    replies: dict[str, list[int]] = defaultdict(list)
    for index, message in enumerate(messages):
        for token in token_sets[index]:
            postings[token].append(index)
        if message.reply_to:
            replies[message.reply_to].append(message.id)
    id_to_index = {message.id: index for index, message in enumerate(messages)}
    return token_sets, postings, replies, id_to_index


def _run_agent(shortlist, query: ScaleQuery, config: Mapping[str, Any]):
    base_tokens, window, counter = _base_tokens(config, query, 160)
    agent = MemoryAgent(shortlist.messages, config)
    return agent.before_response(
        "c1", query.question, _recent_for(query), window=window, output_reserve=4096
    )


def _stage(shortlist, board: bool, final: bool, expected: Sequence[str]) -> str:
    if not set(expected) <= shortlist.ids:
        return "corte_candidatos"
    if not board:
        return "quadro"
    if not final:
        return "orcamento"
    return "ok"


def _evaluate(
    messages: Sequence[Message],
    index,
    queries: Sequence[ScaleQuery],
    group: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    token_sets, postings, replies, id_to_index = index
    rows = []
    for query in queries:
        context = [turn.text for turn in _recent_for(query)]
        before_short = build_shortlist(
            messages,
            query,
            200,
            postings,
            replies,
            id_to_index,
            extra_context=context,
        )
        after_short = build_shortlist(
            messages,
            query,
            200,
            postings,
            replies,
            id_to_index,
            extra_context=context,
            correction=True,
            token_sets=token_sets,
        )
        before_turn = _run_agent(before_short, query, config)
        after_turn = _run_agent(after_short, query, config)
        before_board, before_final = _coverage(before_turn, query.expected_ids)
        after_board, after_final = _coverage(after_turn, query.expected_ids)
        application = after_short.correction
        activated = bool(application and application.activated)
        rows.append(
            {
                "query": query.id,
                "group": group,
                "role": GOLD_ROLES.get(query.id, "regressão conhecida"),
                "expected": list(query.expected_ids),
                "candidate_before": set(query.expected_ids) <= before_short.ids,
                "candidate_after": set(query.expected_ids) <= after_short.ids,
                "board_before": before_board,
                "board_after": after_board,
                "final_before": before_final,
                "final_after": after_final,
                "improved": after_final and not before_final,
                "regressed": before_final and not after_final,
                "stage_before": _stage(
                    before_short, before_board, before_final, query.expected_ids
                ),
                "stage_after": _stage(
                    after_short, after_board, after_final, query.expected_ids
                ),
                "points_before": len(before_short.messages),
                "points_after": len(after_short.messages),
                "search_ms_before": round(before_short.elapsed_ms, 2),
                "search_ms_after": round(after_short.elapsed_ms, 2),
                "memory_tokens_before": before_turn.budget.diagnostics.memory_tokens,
                "memory_tokens_after": after_turn.budget.diagnostics.memory_tokens,
                "correction": (
                    {
                        "activated": application.activated,
                        "reason": application.reason,
                        "entity_terms": list(application.entity_terms),
                        "decision_terms": list(application.decision_terms),
                        "matched": application.matched,
                        "injected": list(application.injected),
                        "displaced": list(application.displaced),
                        "elapsed_ms": round(application.elapsed_ms, 2),
                    }
                    if application
                    else None
                ),
                "activated_without_gain": activated
                and after_final == before_final,
            }
        )
    return rows


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    activated = [row for row in rows if row["correction"]["activated"]]
    return {
        "queries": count,
        "candidate_before": sum(row["candidate_before"] for row in rows),
        "candidate_after": sum(row["candidate_after"] for row in rows),
        "board_before": sum(row["board_before"] for row in rows),
        "board_after": sum(row["board_after"] for row in rows),
        "final_before": sum(row["final_before"] for row in rows),
        "final_after": sum(row["final_after"] for row in rows),
        "improved": sum(row["improved"] for row in rows),
        "regressed": sum(row["regressed"] for row in rows),
        "activated": len(activated),
        "activated_without_gain": sum(
            row["activated_without_gain"] for row in rows
        ),
        "sensitivity_blocked": sum(
            row["correction"]["reason"] == "sensibilidade" for row in rows
        ),
        "injected": sum(
            len(row["correction"]["injected"]) for row in rows
        ),
        "displaced": sum(
            len(row["correction"]["displaced"]) for row in rows
        ),
        "matched_avg": round(
            sum(row["correction"]["matched"] for row in activated)
            / len(activated),
            1,
        )
        if activated
        else 0.0,
        "points_before": round(
            sum(row["points_before"] for row in rows) / count, 1
        ),
        "points_after": round(
            sum(row["points_after"] for row in rows) / count, 1
        ),
        "search_med_before": _percentile(
            [row["search_ms_before"] for row in rows], 0.5
        ),
        "search_med_after": _percentile(
            [row["search_ms_after"] for row in rows], 0.5
        ),
        "search_p95_before": _percentile(
            [row["search_ms_before"] for row in rows], 0.95
        ),
        "search_p95_after": _percentile(
            [row["search_ms_after"] for row in rows], 0.95
        ),
        "correction_med": _percentile(
            [row["correction"]["elapsed_ms"] for row in activated], 0.5
        ),
        "correction_p95": _percentile(
            [row["correction"]["elapsed_ms"] for row in activated], 0.95
        ),
        "memory_before": round(
            sum(row["memory_tokens_before"] for row in rows) / count, 1
        ),
        "memory_after": round(
            sum(row["memory_tokens_after"] for row in rows) / count, 1
        ),
    }


def run_correction_round(
    scale: int = 100_000,
    seed: int = 42,
    candidate_k: int = 200,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    known_messages = generate_messages(scale, seed)
    extended_targets = target_messages(extended=True)
    extended_messages = generate_messages(scale, seed, targets=extended_targets)

    known_rows = _evaluate(
        known_messages,
        _index(known_messages),
        KNOWN_QUERIES,
        "regression",
        config,
    )
    new_rows = _evaluate(
        extended_messages,
        _index(extended_messages),
        NEW_FAMILIES,
        "new",
        config,
    )
    rows = known_rows + new_rows
    traces = [
        {
            "query": row["query"],
            "expected": row["expected"],
            "stage_before": row["stage_before"],
            "stage_after": row["stage_after"],
            "candidate_before": row["candidate_before"],
            "candidate_after": row["candidate_after"],
            "board_before": row["board_before"],
            "board_after": row["board_after"],
            "final_before": row["final_before"],
            "final_after": row["final_after"],
            "correction": row["correction"],
        }
        for row in rows
        if row["stage_before"] != "ok" or row["stage_after"] != "ok"
    ]
    t5 = next(
        (
            row
            for row in known_rows
            if row["query"] == "correcao" and "t5" in row["expected"]
        ),
        None,
    )
    return {
        "scale": scale,
        "seed": seed,
        "candidate_k": candidate_k,
        "budget": 160,
        "messages": len(known_messages),
        "messages_extended": len(extended_messages),
        "gold": {
            row["query"]: row["expected"]
            for row in rows
        },
        "rows": rows,
        "traces": traces,
        "t5": t5,
        "aggregates": {
            "regression": _aggregate(known_rows),
            "new": _aggregate(new_rows),
        },
    }
