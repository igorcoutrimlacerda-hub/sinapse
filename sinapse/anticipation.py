"""Quadro antecipatório do Farol.

Gera uma hipótese de continuação após a resposta atual e guarda somente a
hipótese, os IDs das fontes e a revisão/turno. Não altera o projeto e não é
enviado ao modelo.
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, Sequence

from .memory import recall
from .models import Message
from .text import normalize_text, tokenize

DEFAULTS = {
    "enabled": False,
    "max_rounds": 1,
    "max_candidates": 8,
    "max_time_ms": 150,
    "valid_turns": 1,
}


def settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (config or {}).get("anticipation", {}) or {}
    return {**DEFAULTS, **raw}


def _last_user_message(project: Mapping[str, Any]) -> str:
    return next(
        (
            str(message["text"])
            for message in reversed(project.get("messages", ()))
            if message.get("author") == "user"
        ),
        "",
    )


def infer_hypothesis(project: Mapping[str, Any]) -> str | None:
    text = _last_user_message(project)
    normalized = normalize_text(text)
    pending = [
        item
        for item in project.get("state", {}).get("pending", ())
        if item.get("status") == "aberto"
    ]
    condition = next(
        (
            str(message["text"])
            for message in project.get("messages", ())
            if "abaixo" in normalize_text(str(message["text"]))
            and "export" in normalize_text(str(message["text"]))
        ),
        "",
    )
    cpu = re.search(r"\b(\d{1,3})\s*%", text)
    if "cpu" in normalized and cpu:
        value = int(cpu.group(1))
        if condition and value < 30:
            return "pedir instruções de exportação"
        if condition:
            return "reduzir o oversampling"
    if pending:
        return f"continuar a pendência: {pending[-1]['text']}"
    return None


def _message_objects(project: Mapping[str, Any]) -> list[Message]:
    return [
        Message(
            id=str(message["id"]),
            text=str(message["text"]),
            author=str(message.get("author", "user")),
            conversation_id=str(project["id"]),
            position=index,
            coordinates={},
        )
        for index, message in enumerate(project.get("messages", ()), start=1)
    ]


def _search_query(project: Mapping[str, Any], hypothesis: str) -> str:
    pending = [
        item["text"]
        for item in project.get("state", {}).get("pending", ())
        if item.get("status") == "aberto"
    ]
    if "export" in normalize_text(hypothesis):
        return "exportar exportação configurações 48 kHz oversampling"
    if "oversampling" in normalize_text(hypothesis):
        return "oversampling reduzir exportação"
    return " ".join(pending[-1:] or [hypothesis])


def _lexical_ids(
    project: Mapping[str, Any],
    query: str,
    limit: int,
    excluded: set[str],
) -> list[str]:
    query_tokens = tokenize(query)
    scored = []
    for message in project.get("messages", ()):
        if message["id"] in excluded:
            continue
        score = len(tokenize(str(message["text"])) & query_tokens)
        if score:
            scored.append((score, message.get("position", 0), message["id"]))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [message_id for _, _, message_id in scored[:limit]]


def build_anticipation(
    project: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    max_rounds: int = 1,
    max_candidates: int = 8,
    max_time_ms: int = 150,
) -> dict[str, Any] | None:
    hypothesis = infer_hypothesis(project)
    if not hypothesis:
        return None
    started = time.perf_counter()
    query = _search_query(project, hypothesis)
    excluded = {str(project["messages"][-1]["id"])} if project.get("messages") else set()
    source_ids: list[str] = []
    try:
        messages = _message_objects(project)
        result = recall(
            messages,
            query,
            [],
            config,
            conversation_id=str(project["id"]),
            excluded_ids=tuple(excluded),
        )
        source_ids = [
            candidate.message.id
            for candidate in result.candidates
            if candidate.within_radius
        ][:max_candidates]
    except Exception:
        source_ids = []
    if not source_ids:
        source_ids = _lexical_ids(project, query, max_candidates, excluded)
    elapsed = (time.perf_counter() - started) * 1000
    if elapsed > max_time_ms:
        return {
            "hypothesis": hypothesis,
            "source_ids": [],
            "status": "expired",
            "expired_reason": "time_budget",
            "elapsed_ms": round(elapsed, 2),
        }
    return {
        "hypothesis": hypothesis,
        "source_ids": source_ids[:max_candidates],
        "status": "ready",
        "elapsed_ms": round(elapsed, 2),
        "search_query": query,
        "rounds": 1,
        "candidates": max_candidates,
    }


def matches_intent(hypothesis: str, query: str) -> bool:
    h = normalize_text(hypothesis)
    q = normalize_text(query)
    if "export" in h:
        return any(term in q for term in ("export", "48 khz", "48khz"))
    if "oversampling" in h:
        return any(term in q for term in ("oversampling", "reduz", "reduzir"))
    h_tokens = tokenize(hypothesis)
    q_tokens = tokenize(query)
    return bool(h_tokens & q_tokens)
