"""Modelos de dados da Sinapse."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .text import DIMENSIONS, ReferenceResolution


@dataclass(frozen=True)
class Message:
    """Uma mensagem é um único ponto. Nunca é dividida pelas dimensões."""

    id: str
    text: str
    author: str
    conversation_id: str
    position: int
    sent_at: str | None = None
    reply_to: str | None = None
    coordinates: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def values(self, dimension: str) -> tuple[str, ...]:
        return tuple(self.coordinates.get(dimension, ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "conversation_id": self.conversation_id,
            "position": self.position,
            "sent_at": self.sent_at,
            "reply_to": self.reply_to,
            "coordinates": {d: list(self.values(d)) for d in DIMENSIONS},
        }


@dataclass(frozen=True)
class Candidate:
    message: Message
    distance: float
    association: float
    association_content: float
    association_link: float
    association_source: str
    evidence: Mapping[str, float | None]
    attention: float
    within_radius: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.message.id,
            "text": self.message.text,
            "distance": self.distance,
            "association": self.association,
            "association_content": self.association_content,
            "association_link": self.association_link,
            "association_source": self.association_source,
            "evidence": {d: self.evidence.get(d) for d in DIMENSIONS},
            "attention": self.attention,
            "within_radius": self.within_radius,
        }


@dataclass(frozen=True)
class RecallResult:
    seed: Message
    focus: str
    focus_rule: str
    weights: Mapping[str, float]
    radius: float
    references: ReferenceResolution
    anchor_tokens: frozenset[str]
    candidates: tuple[Candidate, ...]
    evidence: tuple[Candidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed.as_dict(),
            "focus": self.focus,
            "focus_rule": self.focus_rule,
            "weights": dict(self.weights),
            "radius": self.radius,
            "references": self.references.as_dict(),
            "anchor_tokens": sorted(self.anchor_tokens),
            "candidates": [c.as_dict() for c in self.candidates],
            "evidence": [c.message.id for c in self.evidence],
        }


@dataclass(frozen=True)
class Group:
    id: str
    dimension: str
    message_ids: tuple[str, ...]
    label: str
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimension": self.dimension,
            "message_ids": list(self.message_ids),
            "label": self.label,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class GroupIndex:
    groups: tuple[Group, ...]
    threshold: float
    by_message: Mapping[str, tuple[Group, ...]]

    def groups_for(self, message_id: str) -> tuple[Group, ...]:
        return tuple(self.by_message.get(message_id, ()))

    def by_id(self, group_id: str) -> Group | None:
        for group in self.groups:
            if group.id == group_id:
                return group
        return None


def _normalize_coordinates(raw: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    raw = raw or {}
    coordinates: dict[str, tuple[str, ...]] = {}
    for dimension in DIMENSIONS:
        values = raw.get(dimension, ())
        if values is None:
            values = ()
        if isinstance(values, str):
            values = (values,)
        coordinates[dimension] = tuple(str(v) for v in values)
    return coordinates


def load_messages(path: str | Path) -> list[Message]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_messages = data["messages"] if isinstance(data, dict) else data
    messages: list[Message] = []
    for raw in raw_messages:
        messages.append(
            Message(
                id=str(raw["id"]),
                text=str(raw["text"]),
                author=str(raw.get("author", "user")),
                conversation_id=str(raw.get("conversation_id", "c1")),
                position=int(raw.get("position", len(messages) + 1)),
                sent_at=raw.get("sent_at"),
                reply_to=raw.get("reply_to"),
                coordinates=_normalize_coordinates(raw.get("coordinates")),
            )
        )
    ids = [m.id for m in messages]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"IDs duplicados: {duplicated}")
    return messages


def load_demo(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def messages_by_id(messages: Sequence[Message]) -> dict[str, Message]:
    return {m.id: m for m in messages}
