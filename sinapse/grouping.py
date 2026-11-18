"""Grupos sobrepostos por similaridade dentro de cada dimensão.

Regra: para cada dimensão, cada mensagem com valores vira um nó. Dois nós são
ligados quando o Jaccard dos tokens dos valores é >= limiar. Grupos são os
componentes conexos (single-linkage). Como os grupos são construídos por
dimensão, uma mensagem pertence a vários grupos ao mesmo tempo; os grupos de
dimensões diferentes se sobrepõem. O limiar é global e ajustável; valores mais
altos tornam os grupos mais estritos, valores mais baixos fundem mais nós.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from .models import Group, GroupIndex, Message
from .text import DIMENSIONS, jaccard, tokenize_values


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_groups(
    messages: Sequence[Message],
    threshold: float = 0.33,
    thresholds: Mapping[str, float] | None = None,
) -> GroupIndex:
    ordered = sorted(messages, key=lambda m: (m.position, m.id))
    groups: list[Group] = []
    by_message: dict[str, list[Group]] = defaultdict(list)

    for dimension in DIMENSIONS:
        dimension_threshold = float(thresholds.get(dimension, threshold)) if thresholds else float(threshold)
        token_sets = {
            message.id: tokenize_values(message.values(dimension))
            for message in ordered
        }
        members = [message.id for message in ordered if token_sets[message.id]]
        if not members:
            continue
        union_find = _UnionFind(members)
        for i, left_id in enumerate(members):
            for right_id in members[i + 1:]:
                if jaccard(token_sets[left_id], token_sets[right_id]) >= dimension_threshold:
                    union_find.union(left_id, right_id)

        components: dict[str, list[str]] = defaultdict(list)
        for member in members:
            components[union_find.find(member)].append(member)

        ordered_components = sorted(
            components.values(),
            key=lambda ids: min(
                next(m.position for m in ordered if m.id == member_id)
                for member_id in ids
            ),
        )
        for index, component_ids in enumerate(ordered_components, start=1):
            first_message = next(m for m in ordered if m.id == component_ids[0])
            first_value = first_message.values(dimension)[0]
            group = Group(
                id=f"g:{dimension}:{index}",
                dimension=dimension,
                message_ids=tuple(component_ids),
                label=first_value,
                threshold=dimension_threshold,
            )
            groups.append(group)
            for message_id in component_ids:
                by_message[message_id].append(group)

    return GroupIndex(
        groups=tuple(groups),
        threshold=float(threshold),
        by_message={message_id: tuple(items) for message_id, items in by_message.items()},
    )
