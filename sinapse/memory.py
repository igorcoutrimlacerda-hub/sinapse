"""Evocação: semente, distância, associação, evidência e atenção."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .config import load_config
from .models import Candidate, Message, RecallResult
from .text import (
    DIMENSIONS,
    ReferenceResolution,
    cosine_overlap,
    infer_dimension,
    jaccard,
    query_anchor,
    resolve_references,
    tokenize,
    tokenize_values,
)


def message_tokens(message: Message) -> frozenset[str]:
    tokens: set[str] = set(tokenize(message.text))
    for dimension in DIMENSIONS:
        tokens.update(tokenize_values(message.values(dimension)))
    return frozenset(tokens)


def link_between(left: Message, right: Message) -> bool:
    return left.reply_to == right.id or right.reply_to == left.id


def dimension_similarity(
    message: Message,
    seed: Message,
    dimension: str,
    use_links: bool = True,
) -> float:
    """Similaridade de uma dimensão.

    Sem vínculo, ausência de um lado vale 0 (nunca correspondência).
    Com vínculo direto de resposta, o lado sem valor herda o outro apenas
    para este par, pois o vínculo declara o mesmo acontecimento.
    """
    own = tokenize_values(message.values(dimension))
    seed_values = tokenize_values(seed.values(dimension))

    if use_links and link_between(message, seed):
        own_effective = own or seed_values
        seed_effective = seed_values or own
        return jaccard(own_effective, seed_effective)

    return jaccard(own, seed_values)


def weighted_distance(
    message: Message,
    seed: Message,
    weights: Mapping[str, float],
    use_links: bool = True,
) -> float:
    if message.id == seed.id:
        return 0.0
    total = sum(max(0.0, float(w)) for w in weights.values())
    if total <= 0:
        return 1.0
    accumulated = 0.0
    for dimension in DIMENSIONS:
        weight = max(0.0, float(weights.get(dimension, 0.0)))
        if weight <= 0:
            continue
        accumulated += weight * dimension_similarity(message, seed, dimension, use_links)
    return 1.0 - accumulated / total


def content_association(message: Message, anchor: Iterable[str]) -> float:
    return cosine_overlap(message_tokens(message), anchor)


def association(
    message: Message,
    seed: Message,
    anchor: Iterable[str],
    link_floor: float = 0.90,
    use_links: bool = True,
) -> tuple[float, float, float, str]:
    """Retorna (associação, parte textual, parte de vínculo, origem)."""
    content = content_association(message, anchor)
    linked = use_links and link_between(message, seed)
    link_part = float(link_floor) if linked else 0.0
    value = max(content, link_part)
    if link_part > content + 1e-12:
        source = "link"
    elif content > 0:
        source = "content"
    else:
        source = "none"
    return value, content, link_part, source


def evidence_vector(
    message: Message,
    seed: Message,
    focus: str,
    use_links: bool = True,
) -> dict[str, float | None]:
    """Vetor de evidência.

    null: a mensagem não tem anotação própria na dimensão.
    0..1: similaridade entre o conteúdo próprio e o evento da semente.
    O vínculo só fornece contexto para o evento; não cria evidência própria.
    """
    linked = use_links and link_between(message, seed)
    vector: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        own = tokenize_values(message.values(dimension))
        if not own:
            vector[dimension] = None
            continue
        event = tokenize_values(seed.values(dimension))
        if not event and linked:
            # A resposta vinculada fornece o valor que a semente não tinha.
            event = own
        vector[dimension] = jaccard(own, event)
    return vector


def attention_for(
    association_value: float,
    evidence: Mapping[str, float | None],
    focus: str,
) -> float:
    evidence_value = evidence.get(focus)
    if evidence_value is None:
        return 0.0
    return float(association_value) * float(evidence_value)


def requires_seed(message: Message, seed: Message) -> bool:
    """A resposta curta depende da semente para identificar o assunto?

    Regra geral: vínculo direto + nenhum token do o_que da semente no conteúdo
    próprio da mensagem. m2 ("licença") depende de m1 ("Serum 2"/"bass").
    """
    if not link_between(message, seed):
        return False
    subject_tokens = tokenize_values(seed.values("o_que"))
    if not subject_tokens:
        return False
    return not bool(subject_tokens & message_tokens(message))


def seed_score(message: Message, anchor: Iterable[str]) -> float:
    return cosine_overlap(message_tokens(message), anchor)


def choose_seed(
    messages: Sequence[Message],
    anchor: Iterable[str],
    conversation_id: str | None = None,
    seed_bias: Mapping[str, float] | None = None,
    excluded_ids: Sequence[str] = (),
) -> Message:
    excluded = set(excluded_ids)
    pool = [
        m
        for m in messages
        if (conversation_id is None or m.conversation_id == conversation_id)
        and m.id not in excluded
    ]
    if not pool:
        # Todas as mensagens da conversa estão restritas: devolve a melhor
        # disponível apenas para o orçamento; o agente não a envia.
        pool = [m for m in messages if conversation_id is None or m.conversation_id == conversation_id]
    if not pool:
        raise ValueError("Nenhuma mensagem disponível para escolher a semente")
    bias = seed_bias or {}
    return sorted(
        pool,
        key=lambda m: (
            -seed_score(m, anchor) * float(bias.get(m.id, 1.0)),
            m.position,
            m.id,
        ),
    )[0]


def recall(
    messages: Sequence[Message],
    query_text: str,
    recent_turns: Sequence[str] = (),
    config: Mapping | None = None,
    seed_id: str | None = None,
    focus: str | None = None,
    radius: float | None = None,
    use_links: bool = True,
    conversation_id: str | None = None,
    board_context: Sequence[str] = (),
    seed_bias: Mapping[str, float] | None = None,
    anchor_turns: Sequence[str] | None = None,
    excluded_ids: Sequence[str] = (),
) -> RecallResult:
    config = dict(config or load_config())
    anchor_source = list(recent_turns) if anchor_turns is None else list(anchor_turns)
    anchor = query_anchor(query_text, anchor_source + list(board_context))
    excluded = set(excluded_ids)
    seed = (
        next((m for m in messages if m.id == seed_id), None)
        if seed_id is not None
        else choose_seed(messages, anchor, conversation_id, seed_bias, excluded_ids)
    )
    if seed is None:
        raise ValueError(f"Semente não encontrada: {seed_id}")

    inferred_focus, focus_rule = infer_dimension(query_text)
    focus = focus or inferred_focus
    if focus not in DIMENSIONS:
        raise ValueError(f"Dimensão desconhecida: {focus}")

    weights = {d: float(config["base_weights"].get(d, 0.0)) for d in DIMENSIONS}
    weights[focus] = float(config.get("focus_weight", 1.0))
    effective_radius = float(radius if radius is not None else config.get("radius", 0.70))
    link_floor = float(config.get("link_association_floor", 0.90))
    threshold = float(config.get("attention_threshold", 0.05))

    candidates: list[Candidate] = []
    for message in sorted(messages, key=lambda m: (m.position, m.id)):
        if message.id == seed.id:
            continue
        if message.id in excluded:
            continue
        if message.conversation_id != seed.conversation_id:
            continue
        distance = weighted_distance(message, seed, weights, use_links)
        assoc, content, link_part, source = association(
            message, seed, anchor, link_floor, use_links
        )
        evidence = evidence_vector(message, seed, focus, use_links)
        candidates.append(
            Candidate(
                message=message,
                distance=distance,
                association=assoc,
                association_content=content,
                association_link=link_part,
                association_source=source,
                evidence=evidence,
                attention=attention_for(assoc, evidence, focus),
                within_radius=distance <= effective_radius,
            )
        )

    ranked = sorted(
        (
            c
            for c in candidates
            if c.within_radius
            and c.evidence.get(focus) is not None
            and c.attention > threshold
        ),
        key=lambda c: (-c.attention, c.distance, c.message.position, c.message.id),
    )
    references = resolve_references(query_text, recent_turns)
    return RecallResult(
        seed=seed,
        focus=focus,
        focus_rule=focus_rule,
        weights=weights,
        radius=effective_radius,
        references=references,
        anchor_tokens=anchor,
        candidates=tuple(candidates),
        evidence=tuple(ranked),
    )
