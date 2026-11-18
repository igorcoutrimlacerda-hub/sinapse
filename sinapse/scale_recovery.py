"""Rastreia a perda de evidências no benchmark de 100k e testa uma melhoria.

Reproduz exatamente escala=100000, semente=42, candidate_k=200 e orçamento=160.
A melhoria é expandir os tokens de busca com o histórico recente, mantendo o
seletor congelado, o mesmo limite de candidatos, o mesmo renderizador e o mesmo
orçamento.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
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
)
from .text import tokenize
from .whiteboard import MemoryAgent

NEW_QUERIES = (
    ScaleQuery(
        "novo_quantidade_aurora",
        "Quantos presets tem a Aurora?",
        ("Falamos da Aurora.",),
        ("t7",),
        (),
    ),
    ScaleQuery(
        "novo_quantidade_pack",
        "E o pack da Aurora?",
        ("Falamos do pack da Aurora.",),
        ("t8",),
        (),
    ),
    ScaleQuery(
        "novo_preco_curto",
        "E quanto custou?",
        ("Falamos da Aurora.",),
        ("t6",),
        (),
    ),
)


def _rank_before(
    expected_index: int,
    scores: Counter[int],
    messages: Sequence[Message],
) -> int | None:
    if expected_index not in scores:
        return None
    score = scores[expected_index]
    position = messages[expected_index].position
    rank = 1
    for index, value in scores.items():
        if value > score or (
            value == score and messages[index].position > position
        ):
            rank += 1
    return rank


def _source(
    expected_index: int,
    direct: Sequence[int],
    replies: Mapping[str, Sequence[int]],
    messages: Sequence[Message],
) -> str:
    if expected_index in direct:
        return "lexical_direto"
    for index in direct:
        if messages[index].reply_to == messages[expected_index].id:
            return "vinculo_pai"
    for index in direct:
        if messages[expected_index].id in replies.get(messages[index].id, ()):
            return "vinculo_resposta"
    return "nao_encontrado"


def _run_agent(shortlist, query: ScaleQuery, config: Mapping[str, Any]):
    base_tokens, window, counter = _base_tokens(config, query, 160)
    recent = _recent_for(query)
    agent = MemoryAgent(shortlist.messages, config)
    return agent.before_response(
        "c1", query.question, recent, window=window, output_reserve=4096
    )


def _coverage(turn, expected_ids: Sequence[str]) -> tuple[bool, bool]:
    final = all(_id_in_messages(message_id, turn.messages) for message_id in expected_ids)
    board_ids = {item.message_id for item in turn.board}
    board = all(
        message_id in board_ids or _id_in_messages(message_id, turn.messages)
        for message_id in expected_ids
    )
    return board, final


def run_recovery(
    scale: int = 100_000,
    seed: int = 42,
    candidate_k: int = 200,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    messages = generate_messages(scale, seed)
    token_cache = {message.id: _tokens(message) for message in messages}
    postings: dict[str, list[int]] = defaultdict(list)
    replies: dict[str, list[str]] = defaultdict(list)
    for index, message in enumerate(messages):
        for token in token_cache[message.id]:
            postings[token].append(index)
        if message.reply_to:
            replies[message.reply_to].append(message.id)
    id_to_index = {message.id: index for index, message in enumerate(messages)}

    traces = []
    for query in QUERIES:
        if not query.expected_ids:
            continue
        shortlist = build_shortlist(
            messages, query, candidate_k, postings, replies, id_to_index
        )
        turn = _run_agent(shortlist, query, config)
        board, final = _coverage(turn, query.expected_ids)
        scores: Counter[int] = Counter()
        for token in tokenize(query.question):
            for index in postings.get(token, ()):
                scores[index] += 1
        direct_scores = Counter(scores)
        top = heapq.nlargest(
            candidate_k,
            direct_scores.items(),
            key=lambda item: (item[1], messages[item[0]].position),
        )
        direct = [index for index, _ in top]
        selected = [id_to_index[message.id] for message in shortlist.messages]
        for expected in query.expected_ids:
            index = id_to_index.get(expected)
            if index is None:
                continue
            source = _source(index, direct, replies, messages)
            parent_id = messages[index].reply_to
            parent_in_shortlist = bool(parent_id and parent_id in shortlist.ids)
            link_expansion_reached = source in {"vinculo_pai", "vinculo_resposta"}
            after = selected.index(index) + 1 if expected in shortlist.ids else None
            if not final:
                stage = (
                    "corte_candidatos"
                    if source == "nao_encontrado"
                    else "quadro"
                    if not board
                    else "orcamento"
                )
                traces.append(
                    {
                        "query": query.id,
                        "expected": expected,
                        "indexed": token_cache[expected] != frozenset(),
                        "source": source,
                        "expected_reply_to": parent_id,
                        "parent_in_shortlist": parent_in_shortlist,
                        "link_expansion_reached": link_expansion_reached,
                        "rank_before_cut": _rank_before(index, direct_scores, messages),
                        "position_after_cut_and_expansion": after,
                        "in_shortlist": expected in shortlist.ids,
                        "in_board": board,
                        "in_final": final,
                        "first_loss_stage": stage,
                    }
                )

    # Melhoria: tokens do histórico recente na busca de candidatos.
    rows = []
    for query in (*QUERIES, *NEW_QUERIES):
        if not query.expected_ids:
            continue
        before_short = build_shortlist(
            messages, query, candidate_k, postings, replies, id_to_index
        )
        after_short = build_shortlist(
            messages,
            query,
            candidate_k,
            postings,
            replies,
            id_to_index,
            extra_context=[turn.text for turn in _recent_for(query)],
        )
        before_turn = _run_agent(before_short, query, config)
        after_turn = _run_agent(after_short, query, config)
        before_board, before_final = _coverage(before_turn, query.expected_ids)
        after_board, after_final = _coverage(after_turn, query.expected_ids)
        rows.append(
            {
                "query": query.id,
                "group": "regression" if query in QUERIES else "new",
                "expected": list(query.expected_ids),
                "candidate_before": set(query.expected_ids) <= before_short.ids,
                "candidate_after": set(query.expected_ids) <= after_short.ids,
                "board_before": before_board,
                "board_after": after_board,
                "final_before": before_final,
                "final_after": after_final,
                "points_before": len(before_short.messages),
                "points_after": len(after_short.messages),
                "search_ms_before": round(before_short.elapsed_ms, 2),
                "search_ms_after": round(after_short.elapsed_ms, 2),
                "memory_tokens_before": before_turn.budget.diagnostics.memory_tokens,
                "memory_tokens_after": after_turn.budget.diagnostics.memory_tokens,
                "improved": after_final and not before_final,
                "regressed": before_final and not after_final,
            }
        )

    def aggregate(group: str) -> dict[str, Any]:
        subset = [
            row
            for row in rows
            if row["group"] == group and row["expected"]
        ]
        return {
            "queries": len(subset),
            "candidate_before": sum(row["candidate_before"] for row in subset),
            "candidate_after": sum(row["candidate_after"] for row in subset),
            "board_before": sum(row["board_before"] for row in subset),
            "board_after": sum(row["board_after"] for row in subset),
            "final_before": sum(row["final_before"] for row in subset),
            "final_after": sum(row["final_after"] for row in subset),
            "improved": sum(row["improved"] for row in subset),
            "regressed": sum(row["regressed"] for row in subset),
            "points_before": round(
                sum(row["points_before"] for row in subset) / len(subset), 1
            ),
            "points_after": round(
                sum(row["points_after"] for row in subset) / len(subset), 1
            ),
            "search_med_before": _percentile(
                [row["search_ms_before"] for row in subset], 0.5
            ),
            "search_med_after": _percentile(
                [row["search_ms_after"] for row in subset], 0.5
            ),
            "search_sum_before": round(
                sum(row["search_ms_before"] for row in subset), 2
            ),
            "search_sum_after": round(
                sum(row["search_ms_after"] for row in subset), 2
            ),
            "memory_before": round(
                sum(row["memory_tokens_before"] for row in subset) / len(subset), 1
            ),
            "memory_after": round(
                sum(row["memory_tokens_after"] for row in subset) / len(subset), 1
            ),
        }

    return {
        "scale": scale,
        "seed": seed,
        "candidate_k": candidate_k,
        "budget": 160,
        "messages": len(messages),
        "traces": traces,
        "rows": rows,
        "aggregates": {
            "regression": aggregate("regression"),
            "new": aggregate("new"),
        },
    }


def export_recovery(result: Mapping[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "scale_recovery.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)
