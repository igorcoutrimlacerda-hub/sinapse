"""Benchmark longitudinal: preferências e revisões ao longo de semanas.

Compara a política atual congelada (mensagens originais) com uma variante que
registra preferências confirmadas e revisões como entradas derivadas, cada uma
com referência às mensagens originais. A seleção continua sendo o
``MemoryAgent`` congelado; o orçamento de memória é o mesmo nas duas condições.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages
from .config import load_config
from .models import DIMENSIONS, Message, load_messages
from .scale_benchmark import _id_in_messages, _percentile
from .text import normalize_text, tokenize
from .whiteboard import MemoryAgent

DATASET = "benchmark/longitudinal_weeks.json"

PREFERENCE_MARKERS = frozenset({"prefiro", "preferindo", "preferencia"})
REVISION_MARKERS = frozenset({"troquei", "voltei", "mudei", "substitui"})
RESTRICTION_MARKERS = ("nao quero falar", "nao fale sobre", "evite falar")


def _tokens(message: Message) -> frozenset[str]:
    tokens = set(tokenize(message.text))
    for dimension in message.coordinates:
        for value in message.values(dimension):
            tokens.update(tokenize(value))
    return frozenset(tokens)


def _topic(message: Message) -> str:
    values = message.values("para_que")
    return " ".join(values) if values else "geral"


def _pick_value(message: Message, topic_tokens: frozenset[str]) -> str | None:
    for value in message.values("o_que"):
        if not (tokenize(value) & topic_tokens):
            return value
    values = message.values("o_que")
    return values[0] if values else None


@dataclass
class DerivedEntry:
    id: str
    kind: str
    topic: str
    value: str
    status: str
    source_ids: tuple[str, ...]
    date: str
    position: int
    previous_source: str | None = None
    superseded_by: str | None = None

    def render(self) -> str:
        sources = ", ".join(self.source_ids)
        if self.kind == "preferencia":
            if self.status == "vigente":
                return (
                    f"Preferência vigente (hoje, eu prefiro) para {self.topic}: "
                    f"{self.value}. Origem: {sources}."
                )
            return (
                f"Preferência anterior (antes, eu preferia) para {self.topic}: "
                f"{self.value}. Origem: {sources}. Substituída por "
                f"{self.superseded_by or 'preferência vigente'}."
            )
        if self.status == "vigente":
            extra = f" Anterior: {self.previous_source}." if self.previous_source else ""
            return (
                f"Preset vigente (hoje) para {self.topic}: {self.value}. "
                f"Origem: {sources}.{extra}"
            )
        return (
            f"Preset anterior (antes, eu usava) para {self.topic}: {self.value}. "
            f"Origem: {sources}. Substituída por "
            f"{self.superseded_by or 'preset vigente'}."
        )

    def as_message(self) -> Message:
        return Message(
            id=self.id,
            text=self.render(),
            author="memory",
            conversation_id="c1",
            position=self.position,
            sent_at=self.date,
            coordinates={
                "o_que": (self.value,),
                "quem": ("usuário",),
                "para_que": (self.topic,),
                "quando": (self.date,),
            },
        )


def derive_memory(messages: Sequence[Message]) -> list[DerivedEntry]:
    """Extrai preferências confirmadas e revisões com origem explícita."""

    entries: list[DerivedEntry] = []
    last_preference: dict[str, DerivedEntry] = {}
    last_revision: dict[str, DerivedEntry] = {}
    ordered = sorted(messages, key=lambda message: message.position)
    for message in ordered:
        normalized = normalize_text(message.text)
        if any(marker in normalized for marker in RESTRICTION_MARKERS):
            continue
        tokens = _tokens(message)
        topic = _topic(message)
        topic_tokens = tokenize(topic)
        date = (message.sent_at or "")[:10]
        if tokens & PREFERENCE_MARKERS:
            value = _pick_value(message, topic_tokens)
            if value:
                existing = last_preference.get(topic)
                if existing and normalize_text(existing.value) == normalize_text(value):
                    existing.source_ids = existing.source_ids + (message.id,)
                    existing.date = date
                    existing.position = message.position
                else:
                    entry = DerivedEntry(
                        id=f"d:pref:{len(entries) + 1}",
                        kind="preferencia",
                        topic=topic,
                        value=value,
                        status="vigente",
                        source_ids=(message.id,),
                        date=date,
                        position=message.position,
                    )
                    if existing:
                        existing.status = "anterior"
                        existing.superseded_by = entry.id
                    entries.append(entry)
                    last_preference[topic] = entry
        if tokens & REVISION_MARKERS:
            value = _pick_value(message, topic_tokens)
            if value:
                existing = last_revision.get(topic)
                if existing:
                    previous_source = existing.source_ids[0]
                    previous_value = existing.value
                    existing.status = "anterior"
                    existing.superseded_by = f"d:rev:{len(entries) + 1}"
                else:
                    previous_source = None
                    previous_value = None
                    for candidate in reversed(ordered):
                        if candidate.position >= message.position:
                            continue
                        if _topic(candidate) != topic:
                            continue
                        candidate_value = _pick_value(candidate, topic_tokens)
                        if candidate_value and normalize_text(candidate_value) != normalize_text(value):
                            previous_source = candidate.id
                            previous_value = candidate_value
                            break
                entry = DerivedEntry(
                    id=f"d:rev:{len(entries) + 1}",
                    kind="revisao",
                    topic=topic,
                    value=value,
                    status="vigente",
                    source_ids=(message.id,),
                    date=date,
                    position=message.position,
                    previous_source=previous_source,
                )
                if existing:
                    existing.superseded_by = entry.id
                entries.append(entry)
                last_revision[topic] = entry
    return entries


def derived_messages(entries: Sequence[DerivedEntry]) -> list[Message]:
    return [entry.as_message() for entry in entries]


@dataclass(frozen=True)
class LongitudinalQuery:
    id: str
    family: str
    question: str
    recent: tuple[str, ...]
    evidence_groups: tuple[tuple[str, ...], ...]
    old_ids: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


QUERIES = (
    LongitudinalQuery(
        "decisao_atual",
        "decisao",
        "Qual preset está vigente agora para o bass?",
        ("Falamos de presets para o bass.",),
        (("w6a1",),),
        old_ids=("w1a1", "w3a1"),
    ),
    LongitudinalQuery(
        "decisao_anterior",
        "decisao",
        "Qual preset eu usava antes para o bass?",
        ("Falamos de presets para o bass.",),
        (("w3a1",),),
    ),
    LongitudinalQuery(
        "preferencia_atual",
        "preferencia",
        "Que tipo de preset eu prefiro hoje para o bass?",
        ("Falamos de presets para o bass.",),
        (("w5a1", "w7a1"),),
        old_ids=("w1a2", "w2a1", "w4a2"),
    ),
    LongitudinalQuery(
        "preferencia_antiga",
        "preferencia",
        "Que tipo de preset eu preferia no começo para o bass?",
        ("Falamos de presets para o bass.",),
        (("w1a2", "w2a1", "w4a2"),),
    ),
    LongitudinalQuery(
        "retorno_assunto",
        "retorno",
        "Em que BPM ficou a mixagem do bass?",
        ("Falamos da mixagem do bass.",),
        (("w1a3", "w3a3", "w4a1"), ("w6a3", "w7a3")),
    ),
    LongitudinalQuery(
        "detalhe_unico",
        "detalhe",
        "Quantos knobs tem o meu controlador?",
        ("Falamos do controlador.",),
        (("w1a4",),),
    ),
    LongitudinalQuery(
        "sensibilidade",
        "sensibilidade",
        "Por que eu comprei a licença da Aurora?",
        ("Falamos da Aurora.",),
        (),
        preferences=("Não quero falar sobre a licença da Aurora.",),
        forbidden=("w2a2",),
    ),
)


def _window(
    config: Mapping[str, Any], query: LongitudinalQuery
) -> tuple[int, int]:
    counter = EstimatedChatCounter(SimulatedCounter())
    recent = tuple(RecentTurn(text=text) for text in query.recent)
    base = build_base_messages(
        str(config.get("instructions", "")), recent, query.question
    )
    base_tokens = counter.count_messages(base)
    return base_tokens + 4096 + 160, base_tokens


def _family_satisfied(
    groups: Sequence[Sequence[str]],
    selected: Sequence[str],
    selected_derived: Sequence[DerivedEntry],
) -> bool:
    for group in groups:
        if any(origin in selected for origin in group):
            continue
        if any(
            set(entry.source_ids) & set(group) for entry in selected_derived
        ):
            continue
        return False
    return True


def _evaluate(
    messages: Sequence[Message],
    entries: Sequence[DerivedEntry],
    query: LongitudinalQuery,
    config: Mapping[str, Any],
    with_derived: bool,
) -> dict[str, Any]:
    by_id = {entry.id: entry for entry in entries}
    condition_messages = list(messages) + (
        derived_messages(entries) if with_derived else []
    )
    window, base_tokens = _window(config, query)
    started = time.perf_counter()
    agent = MemoryAgent(condition_messages, config)
    for preference in query.preferences:
        agent.declare_preference("c1", preference, 0)
    turn = agent.before_response(
        "c1",
        query.question,
        query.recent,
        window=window,
        output_reserve=4096,
    )
    elapsed = (time.perf_counter() - started) * 1000
    selected = [
        message.id
        for message in condition_messages
        if _id_in_messages(message.id, turn.messages)
    ]
    selected_derived = [
        by_id[message_id]
        for message_id in selected
        if message_id in by_id
    ]
    sources_valid = all(
        source in {message.id for message in messages}
        for entry in selected_derived
        for source in entry.source_ids
    )
    recovered = _family_satisfied(
        query.evidence_groups, selected, selected_derived
    )
    originals_recovered = all(
        any(origin in selected for origin in group)
        for group in query.evidence_groups
    )
    via_derived = all(
        any(
            set(entry.source_ids) & set(group) for entry in selected_derived
        )
        for group in query.evidence_groups
    )
    leaked = [message_id for message_id in query.old_ids if message_id in selected]
    forbidden = [message_id for message_id in query.forbidden if message_id in selected]
    return {
        "query": query.id,
        "family": query.family,
        "condition": "derivada" if with_derived else "referencia",
        "recovered": recovered,
        "originals_recovered": originals_recovered,
        "via_derived": via_derived,
        "selected": selected,
        "selected_derived": [entry.id for entry in selected_derived],
        "derived_previous_selected": [
            entry.id for entry in selected_derived if entry.status == "anterior"
        ],
        "sources_valid": sources_valid,
        "leaked": leaked,
        "forbidden": forbidden,
        "blocked": list(turn.sensitivity.blocked_ids),
        "select_ms": round(elapsed, 2),
        "memory_tokens": turn.budget.diagnostics.memory_tokens,
        "base_tokens": base_tokens,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "queries": count,
        "recovered": sum(row["recovered"] for row in rows),
        "originals_recovered": sum(row["originals_recovered"] for row in rows),
        "via_derived": sum(row["via_derived"] for row in rows),
        "leak_queries": sum(bool(row["leaked"]) for row in rows),
        "forbidden": sum(bool(row["forbidden"]) for row in rows),
        "sources_valid": sum(row["sources_valid"] for row in rows),
        "select_ms_median": _percentile(
            [row["select_ms"] for row in rows], 0.5
        ),
        "select_ms_p95": _percentile(
            [row["select_ms"] for row in rows], 0.95
        ),
        "memory_tokens_avg": round(
            sum(row["memory_tokens"] for row in rows) / count, 1
        ),
    }


def run_longitudinal(
    path: str | Path = DATASET,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    messages = load_messages(path)
    entries = derive_memory(messages)
    rows = []
    for query in QUERIES:
        for with_derived in (False, True):
            rows.append(
                _evaluate(messages, entries, query, config, with_derived)
            )
    by_condition = {
        condition: [
            row for row in rows if row["condition"] == condition
        ]
        for condition in ("referencia", "derivada")
    }
    pairs = []
    for query in QUERIES:
        reference = next(
            row
            for row in by_condition["referencia"]
            if row["query"] == query.id
        )
        derived = next(
            row
            for row in by_condition["derivada"]
            if row["query"] == query.id
        )
        pairs.append(
            {
                "query": query.id,
                "family": query.family,
                "recovered_reference": reference["recovered"],
                "recovered_derived": derived["recovered"],
                "recovered_by_origin_reference": reference[
                    "originals_recovered"
                ],
                "recovered_by_origin_derived": derived[
                    "originals_recovered"
                ],
                "recovered_via_derived": derived["via_derived"],
                "leak_reference": reference["leaked"],
                "leak_derived": derived["leaked"],
                "origin_preserved": derived["sources_valid"],
                "helped": derived["recovered"] and not reference["recovered"],
                "hurt": reference["recovered"] and not derived["recovered"],
                "leak_helped": len(derived["leaked"]) < len(reference["leaked"]),
                "leak_hurt": len(derived["leaked"]) > len(reference["leaked"]),
            }
        )
    return {
        "dataset": str(path),
        "messages": len(messages),
        "weeks": 7,
        "budget": 160,
        "derived_entries": [
            {
                "id": entry.id,
                "kind": entry.kind,
                "status": entry.status,
                "topic": entry.topic,
                "value": entry.value,
                "source_ids": list(entry.source_ids),
                "previous_source": entry.previous_source,
                "superseded_by": entry.superseded_by,
            }
            for entry in entries
        ],
        "rows": rows,
        "pairs": pairs,
        "aggregates": {
            condition: _aggregate(group)
            for condition, group in by_condition.items()
        },
        "helped": sum(pair["helped"] for pair in pairs),
        "hurt": sum(pair["hurt"] for pair in pairs),
        "leak_helped": sum(pair["leak_helped"] for pair in pairs),
        "leak_hurt": sum(pair["leak_hurt"] for pair in pairs),
    }


def export_longitudinal(
    result: Mapping[str, Any], out_dir: str | Path
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "longitudinal_results.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)
