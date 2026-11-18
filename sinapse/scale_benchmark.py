"""Benchmark de escala: 1.000, 10.000 e 100.000 mensagens sintéticas.

Dados fictícios, semente fixa e as mesmas consultas/evidências/orçamento em
todas as escalas. Mede a perda em três etapas:

1. candidatos: a evidência entrou na lista curta?
2. pontuação: passou por raio/evidência/atributo/atenção?
3. contexto final: entrou na requisição?

Compara Sinapse e busca lexical sob o mesmo orçamento. Não chama o DeepSeek.
"""

from __future__ import annotations

import heapq
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attributes import (
    ATTRIBUTE_PRICE,
    attribute_targets,
    requested_attribute,
    supporting_ids as attribute_supporting_ids,
)
from .budget import RecentTurn, SimulatedCounter
from .chat import EstimatedChatCounter, build_base_messages, select_chat_context
from .config import load_config
from .correction_source import CorrectionSelection, find_corrections
from .deepseek import DeepSeekSettings, load_deepseek_settings
from .memory import message_tokens, recall
from .models import DIMENSIONS, Candidate, Message
from .sensitivity import SensitivityPolicy
from .text import cosine_overlap, normalize_text, tokenize
from .whiteboard import MemoryAgent

TARGET_IDS = ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
WRONG_ENTITIES = ("Nebula", "Comet", "Vega", "Orion", "Lyra", "Nova", "Atlas")
OTHER_TOPICS = (
    "Passei a manhã ajustando a iluminação do estúdio.",
    "Li um artigo sobre síntese subtrativa.",
    "Organizei os cabos e limpei a mesa.",
    "Assisti a uma aula de mixagem.",
    "Comparei dois fones de ouvido.",
    "Anotei ideias para a próxima música.",
    "Testei um plugin de reverberação.",
    "Reorganizei a biblioteca de samples.",
)


def target_messages(extended: bool = False) -> list[Message]:
    specs = [
        ("t1", "Escolhi a Aurora para fazer o bass.", None, {"o_que": ("Aurora", "bass"), "quem": ("usuário",), "para_que": ("fazer o bass",)}),
        ("t2", "Porque eu já tinha comprado a licença da Aurora.", "t1", {"quem": ("usuário",), "por_que": ("já tinha comprado a licença",)}),
        ("t3", "Testei a Aurora alternando entre dois envelopes.", "t1", {"o_que": ("Aurora",), "quem": ("usuário",), "como": ("alternar entre dois envelopes",)}),
        ("t4", "Um amigo indicou a Aurora.", "t1", {"o_que": ("Aurora",), "quem": ("amigo",), "por_que": ("indicação de um amigo",)}),
        ("t5", "Troquei a Aurora pelo Pigments.", None, {"o_que": ("Pigments",), "quem": ("usuário",), "por_que": ("troquei de preset",)}),
        ("t6", "Paguei 200 reais na Aurora.", None, {"o_que": ("Aurora",), "quem": ("usuário",), "por_que": ("paguei 200 reais",)}),
        ("t7", "A Aurora tem 8 presets.", None, {"o_que": ("Aurora",), "quem": ("usuário",)}),
        ("t8", "O pack da Aurora tem 12 presets.", None, {"o_que": ("Aurora", "pack"), "quem": ("usuário",)}),
    ]
    if extended:
        specs.extend(
            [
                ("t9", "Troquei o pack da Aurora pelo pack da Nebula.", None, {"o_que": ("pack", "Nebula"), "quem": ("usuário",), "por_que": ("troquei o pack",)}),
                ("t10", "Depois da correção, paguei 250 reais na Aurora.", None, {"o_que": ("Aurora",), "quem": ("usuário",), "por_que": ("paguei 250 reais",)}),
            ]
        )
    messages: list[Message] = []
    for index, (message_id, text, reply_to, coordinates) in enumerate(specs, start=1):
        messages.append(
            Message(
                id=message_id,
                text=text,
                author="user",
                conversation_id="c1",
                position=index,
                sent_at=f"2026-01-01T09:{index:02d}:00-03:00",
                reply_to=reply_to,
                coordinates={dimension: coordinates.get(dimension, ()) for dimension in DIMENSIONS},
            )
        )
    return messages


@dataclass(frozen=True)
class ScaleQuery:
    id: str
    question: str
    recent: tuple[str, ...]
    expected_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    preferences: tuple[str, ...] = ()
    sensitivity_expected: bool = False
    no_answer: bool = False
    recent_ids: tuple[str | None, ...] = ()


QUERIES = (
    ScaleQuery("motivo", "Por que eu escolhi a Aurora?", ("Falamos da Aurora e do bass.",), ("t2",), ("t1",)),
    ScaleQuery("procedimento", "Como eu testei a Aurora?", ("Falamos da Aurora e do bass.",), ("t3",), ("t1",)),
    ScaleQuery("duas_evidencias", "Por que eu escolhi a Aurora?", ("Falamos da Aurora e do bass.",), ("t2", "t4"), ("t1",)),
    ScaleQuery("correcao", "Troquei para qual preset?", ("Escolhi a Aurora para o bass.",), ("t5",), ("t5",)),
    ScaleQuery("preco_presente", "Quanto custou a Aurora?", ("Falamos da Aurora.",), ("t6",), ("t6",)),
    ScaleQuery("preco_ausente", "Quanto custou o Piano V?", ("Falamos do Piano V.",), (), (), no_answer=True),
    ScaleQuery("quantidade", "Quanto custou o pack de presets?", ("Falamos do pack.",), (), (), no_answer=True),
    ScaleQuery("referencia_curta", "E como eu testei?", ("Por que eu escolhi a Aurora?",), ("t3",), ("t1",)),
    ScaleQuery("ambiguidade", "E aquilo que eu troquei?", ("Falamos da Aurora e do Pigments.",), ("t5",), ("t5",)),
    ScaleQuery(
        "resolvida",
        "Por que eu escolhi a Aurora?",
        ("Porque eu já tinha comprado a licença da Aurora.", "Falamos da Aurora."),
        ("t2",),
        (),
        recent_ids=("t2", None),
    ),
    ScaleQuery("sensibilidade", "Por que eu escolhi a Aurora?", ("Falamos da Aurora.",), (), (), preferences=("Não quero falar sobre a licença da Aurora.",), sensitivity_expected=True),
)


def generate_messages(
    scale: int,
    seed: int = 42,
    targets: Sequence[Message] | None = None,
) -> list[Message]:
    rng = random.Random(seed)
    targets = list(targets) if targets is not None else target_messages()
    noise: list[Message] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"n{counter:06d}"

    while len(targets) + len(noise) < scale:
        draw = rng.random()
        if draw < 0.62:
            sentence = rng.choice(OTHER_TOPICS)
            message_id = next_id()
            noise.append(
                Message(
                    id=message_id,
                    text=sentence,
                    author="user",
                    conversation_id="c1",
                    position=0,
                    sent_at="2026-01-02T09:00:00-03:00",
                    coordinates={dimension: () for dimension in DIMENSIONS},
                )
            )
        elif draw < 0.90:
            entity = rng.choice(WRONG_ENTITIES)
            parent_id = next_id()
            noise.append(
                Message(
                    id=parent_id,
                    text=f"Escolhi a {entity} para fazer o bass.",
                    author="user",
                    conversation_id="c1",
                    position=0,
                    sent_at="2026-01-02T09:00:00-03:00",
                    coordinates={
                        "o_que": (entity, "bass"),
                        "quem": ("usuário",),
                        "para_que": ("fazer o bass",),
                    },
                )
            )
            if len(targets) + len(noise) < scale:
                noise.append(
                    Message(
                        id=next_id(),
                        text=f"Porque eu já tinha comprado a licença da {entity}.",
                        author="user",
                        conversation_id="c1",
                        position=0,
                        sent_at="2026-01-02T09:00:00-03:00",
                        reply_to=parent_id,
                        coordinates={
                            "quem": ("usuário",),
                            "por_que": (f"já tinha comprado a licença da {entity}",),
                        },
                    )
                )
        else:
            entity = rng.choice(WRONG_ENTITIES)
            noise.append(
                Message(
                    id=next_id(),
                    text=f"Antes eu usava a Aurora, mas troquei pela {entity}.",
                    author="user",
                    conversation_id="c1",
                    position=0,
                    sent_at="2026-01-02T09:00:00-03:00",
                    coordinates={
                        "o_que": (entity,),
                        "quem": ("usuário",),
                        "por_que": ("troquei de preset",),
                    },
                )
            )

    messages = targets + noise
    rng.shuffle(messages)
    return [
        Message(
            id=message.id,
            text=message.text,
            author=message.author,
            conversation_id=message.conversation_id,
            position=index,
            sent_at=message.sent_at,
            reply_to=message.reply_to,
            coordinates=message.coordinates,
        )
        for index, message in enumerate(messages, start=1)
    ]


def _tokens(message: Message) -> frozenset[str]:
    tokens = set(tokenize(message.text))
    for dimension in message.coordinates:
        for value in message.values(dimension):
            tokens.update(tokenize(value))
    return frozenset(tokens)


@dataclass(frozen=True)
class CorrectionApplication:
    activated: bool
    reason: str
    entity_terms: tuple[str, ...] = ()
    decision_terms: tuple[str, ...] = ()
    matched: int = 0
    injected: tuple[str, ...] = ()
    displaced: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass
class Shortlist:
    messages: list[Message]
    ids: frozenset[str]
    elapsed_ms: float
    correction: CorrectionApplication | None = None


def build_shortlist(
    messages: Sequence[Message],
    query: ScaleQuery,
    candidate_k: int,
    postings: Mapping[str, Sequence[int]],
    replies: Mapping[str, Sequence[int]],
    id_to_index: Mapping[str, int],
    extra_context: Sequence[str] = (),
    correction: bool = False,
    token_sets: Mapping[int, frozenset[str]] | None = None,
) -> Shortlist:
    started = time.perf_counter()
    query_tokens = set(tokenize(query.question))
    for text in extra_context:
        query_tokens.update(tokenize(text))
    attribute = requested_attribute(query.question)
    if attribute == ATTRIBUTE_PRICE:
        query_tokens.update({"paguei", "pagou", "custou", "custa", "reais", "preco", "valor"})
    scores: Counter[int] = Counter()
    for token in query_tokens:
        for index in postings.get(token, ()):
            scores[index] += 1

    baseline_top = heapq.nlargest(
        candidate_k,
        scores.items(),
        key=lambda item: (item[1], messages[item[0]].position),
    )
    baseline_selected = [index for index, _ in baseline_top]
    application = None
    if correction:
        if token_sets is None:
            token_sets = {
                index: _tokens(message) for index, message in enumerate(messages)
            }
        selection: CorrectionSelection = find_corrections(
            messages,
            token_sets,
            postings,
            query.question,
            history=extra_context,
            preferences=query.preferences,
        )
        baseline_ids = set(baseline_selected)
        extra_indices = [
            index for index in selection.indices if index not in baseline_ids
        ]
        cut_scores = scores
        if extra_indices:
            boost = max(scores.values(), default=0) + 1
            cut_scores = Counter(scores)
            for index in extra_indices:
                cut_scores[index] = boost
            baseline_top = heapq.nlargest(
                candidate_k,
                cut_scores.items(),
                key=lambda item: (item[1], messages[item[0]].position),
            )
        selected_after = {index for index, _ in baseline_top}
        application = CorrectionApplication(
            activated=selection.activated,
            reason=selection.reason,
            entity_terms=selection.entity_terms,
            decision_terms=selection.decision_terms,
            matched=selection.matched,
            injected=tuple(
                messages[index].id
                for index in extra_indices
                if index in selected_after
            ),
            displaced=tuple(
                messages[index].id
                for index in baseline_selected
                if index not in selected_after
            ),
            elapsed_ms=selection.elapsed_ms,
        )
        top = baseline_top
    else:
        top = baseline_top
    selected: list[int] = [index for index, _ in top]
    extra: list[int] = []
    for index in selected:
        message = messages[index]
        if message.reply_to:
            parent_index = id_to_index.get(message.reply_to)
            if parent_index is not None and parent_index not in selected:
                extra.append(parent_index)
        for reply_id in replies.get(message.id, ()):
            reply_index = id_to_index.get(reply_id)
            if reply_index is not None and reply_index not in selected:
                extra.append(reply_index)
    for index in extra:
        if index not in selected:
            selected.append(index)
    shortlisted = [messages[index] for index in selected]
    elapsed = (time.perf_counter() - started) * 1000
    return Shortlist(
        messages=shortlisted,
        ids=frozenset(message.id for message in shortlisted),
        elapsed_ms=elapsed,
        correction=application,
    )


def _id_in_messages(message_id: str, messages: Sequence[Mapping[str, str]]) -> bool:
    context = "\n".join(str(message["content"]) for message in messages)
    pattern = rf"(?<![a-z0-9_])id={re.escape(message_id)}(?!\d)"
    return bool(re.search(pattern, context)) or f"[{message_id}]" in context


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 2)


def _score_stage(
    query: ScaleQuery,
    expected_id: str,
    candidates: Mapping[str, Candidate],
    focus: str,
    threshold: float,
    attribute: str | None,
    seed_id: str,
    blocked: set[str],
    targets: frozenset[str] | None = None,
    by_id: Mapping[str, Message] | None = None,
) -> bool:
    if expected_id in blocked:
        return False
    if expected_id == seed_id:
        return True
    candidate = candidates.get(expected_id)
    if candidate is None or not candidate.within_radius:
        return False
    if attribute == ATTRIBUTE_PRICE:
        from .attributes import message_supports_attribute

        return message_supports_attribute(candidate.message, attribute, targets, by_id)
    return candidate.evidence.get(focus) is not None and candidate.attention > threshold


@dataclass
class QueryOutcome:
    query: str
    condition: str
    candidate_ids: frozenset[str]
    scored_ids: frozenset[str]
    final_ids: frozenset[str]
    missing_stage: str | None
    sensitivity_excluded: bool
    no_answer_empty: bool | None
    search_ms: float
    select_ms: float
    points: int
    tokens: int
    extra_candidates: int = 0
    new_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "condition": self.condition,
            "candidate_ids": sorted(self.candidate_ids),
            "scored_ids": sorted(self.scored_ids),
            "final_ids": sorted(self.final_ids),
            "missing_stage": self.missing_stage,
            "sensitivity_excluded": self.sensitivity_excluded,
            "no_answer_empty": self.no_answer_empty,
            "search_ms": round(self.search_ms, 2),
            "select_ms": round(self.select_ms, 2),
            "points": self.points,
            "tokens": self.tokens,
            "extra_candidates": self.extra_candidates,
            "new_ids": list(self.new_ids),
        }


def _candidate_map(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    return {candidate.message.id: candidate for candidate in candidates}


def _recent_for(query: ScaleQuery) -> tuple[RecentTurn, ...]:
    ids = query.recent_ids or (None,) * len(query.recent)
    return tuple(
        RecentTurn(text=text, id=message_id)
        for text, message_id in zip(query.recent, ids)
    )


def _base_tokens(
    config: Mapping[str, Any], query: ScaleQuery, window_limit: int
) -> tuple[int, int, int]:
    counter = EstimatedChatCounter(SimulatedCounter())
    recent = _recent_for(query)
    base = build_base_messages(
        str(config.get("instructions", "")), recent, query.question
    )
    base_tokens = counter.count_messages(base)
    return base_tokens, base_tokens + 4096 + window_limit, counter  # reserve fixa


def run_scale(scale: int, seed: int, candidate_k: int, config: Mapping[str, Any]) -> dict:
    started = time.perf_counter()
    messages = generate_messages(scale, seed)
    generated = time.perf_counter() - started
    token_cache = {message.id: _tokens(message) for message in messages}
    postings: dict[str, list[int]] = defaultdict(list)
    replies: dict[str, list[str]] = defaultdict(list)
    started_index = time.perf_counter()
    for index, message in enumerate(messages):
        for token in token_cache[message.id]:
            postings[token].append(index)
        if message.reply_to:
            replies[message.reply_to].append(message.id)
    index_time = time.perf_counter() - started_index
    by_id = {message.id: message for message in messages}
    id_to_index = {message.id: index for index, message in enumerate(messages)}
    threshold = float(config.get("attention_threshold", 0.05))
    outcomes: list[QueryOutcome] = []

    for query in QUERIES:
        shortlist = build_shortlist(
            messages, query, candidate_k, postings, replies, id_to_index
        )
        base_tokens, window, counter = _base_tokens(config, query, 160)
        recent = _recent_for(query)

        # Sinapse: quadro + temas + sensibilidade + utilidade.
        started_select = time.perf_counter()
        agent = MemoryAgent(shortlist.messages, config)
        for preference in query.preferences:
            agent.declare_preference("c1", preference, 0)
        turn = agent.before_response(
            "c1", query.question, recent, window=window, output_reserve=4096
        )
        select_ms = (time.perf_counter() - started_select) * 1000
        candidates = _candidate_map(turn.recall.candidates)
        attribute = requested_attribute(query.question)
        blocked = set(turn.sensitivity.blocked_ids)
        scored = {
            expected
            for expected in query.expected_ids
            if _score_stage(
                query,
                expected,
                candidates,
                turn.focus,
                threshold,
                attribute,
                turn.seed.id,
                blocked,
                attribute_targets(query.question),
                {message.id: message for message in shortlist.messages},
            )
        }
        final = {expected for expected in query.expected_ids if _id_in_messages(expected, turn.messages)}
        missing = None
        if query.expected_ids:
            if not set(query.expected_ids) <= shortlist.ids:
                missing = "candidates"
            elif not set(query.expected_ids) <= scored:
                missing = "scoring"
            elif not set(query.expected_ids) <= final:
                missing = "selection"
        outcomes.append(
            QueryOutcome(
                query=query.id,
                condition="sinapse",
                candidate_ids=frozenset(
                    expected for expected in query.expected_ids if expected in shortlist.ids
                ),
                scored_ids=frozenset(scored),
                final_ids=frozenset(final),
                missing_stage=missing,
                sensitivity_excluded=query.sensitivity_expected and "t2" in blocked,
                no_answer_empty=(
                    not turn.budget.diagnostics.selected_ids
                    and not turn.budget.diagnostics.seed_included
                    if query.no_answer
                    else None
                ),
                search_ms=shortlist.elapsed_ms,
                select_ms=select_ms,
                points=len(shortlist.messages),
                tokens=turn.budget.diagnostics.used_tokens,
            )
        )

        # Sinapse com curiosidade ligada (mesmo shortlist e orçamento).
        started_select = time.perf_counter()
        curious_agent = MemoryAgent(shortlist.messages, config, curiosity=True)
        for preference in query.preferences:
            curious_agent.declare_preference("c1", preference, 0)
        curious_turn = curious_agent.before_response(
            "c1", query.question, recent, window=window, output_reserve=4096
        )
        curious_select_ms = (time.perf_counter() - started_select) * 1000
        curious_candidates = _candidate_map(curious_turn.recall.candidates)
        curious_blocked = set(curious_turn.sensitivity.blocked_ids)
        curious_scored = {
            expected
            for expected in query.expected_ids
            if _score_stage(
                query,
                expected,
                curious_candidates,
                curious_turn.focus,
                threshold,
                attribute,
                curious_turn.seed.id,
                curious_blocked,
                attribute_targets(query.question),
                {message.id: message for message in shortlist.messages},
            )
        }
        curious_final = {
            expected
            for expected in query.expected_ids
            if _id_in_messages(expected, curious_turn.messages)
        }
        curious_missing = None
        if query.expected_ids:
            if not set(query.expected_ids) <= shortlist.ids:
                curious_missing = "candidates"
            elif not set(query.expected_ids) <= curious_scored:
                curious_missing = "scoring"
            elif not set(query.expected_ids) <= curious_final:
                curious_missing = "selection"
        outcomes.append(
            QueryOutcome(
                query=query.id,
                condition="sinapse_curioso",
                candidate_ids=frozenset(
                    expected for expected in query.expected_ids if expected in shortlist.ids
                ),
                scored_ids=frozenset(curious_scored),
                final_ids=frozenset(curious_final),
                missing_stage=curious_missing,
                sensitivity_excluded=query.sensitivity_expected and "t2" in curious_blocked,
                no_answer_empty=(
                    not curious_turn.budget.diagnostics.selected_ids
                    and not curious_turn.budget.diagnostics.seed_included
                    if query.no_answer
                    else None
                ),
                search_ms=shortlist.elapsed_ms,
                select_ms=curious_select_ms,
                points=len(shortlist.messages),
                tokens=curious_turn.budget.diagnostics.used_tokens,
                extra_candidates=len(curious_turn.curiosity.evaluated_ids),
                new_ids=curious_turn.curiosity.new_in_context,
            )
        )

        # Lexical: mesmo shortlist, mesmo orçamento, sem quadro/tema/utilidade.
        policy = SensitivityPolicy()
        for preference in query.preferences:
            policy.declare("c1", preference, 0)
        blocked_policy = set(
            policy.evaluate("c1", query.question, shortlist.messages, 0).blocked_ids
        )
        started_select = time.perf_counter()
        recall_result = recall(
            shortlist.messages,
            query.question,
            [turn.text for turn in recent],
            config,
            conversation_id="c1",
            excluded_ids=tuple(blocked_policy),
        )
        lexical_candidates: list[Candidate] = []
        query_tokens = tokenize(query.question)
        for message in shortlist.messages:
            if message.id == recall_result.seed.id or message.id in blocked_policy:
                continue
            score = cosine_overlap(message_tokens(message), query_tokens)
            if score <= 0:
                continue
            evidence: dict[str, float | None] = {
                dimension: None for dimension in DIMENSIONS
            }
            evidence[recall_result.focus] = 1.0
            lexical_candidates.append(
                Candidate(
                    message=message,
                    distance=1.0 - score,
                    association=score,
                    association_content=score,
                    association_link=0.0,
                    association_source="lexical",
                    evidence=evidence,
                    attention=score,
                    within_radius=True,
                )
            )
        lexical = select_chat_context(
            instructions=str(config.get("instructions", "")),
            recent_turns=recent,
            current_turn=query.question,
            seed=recall_result.seed,
            focus=recall_result.focus,
            candidates=lexical_candidates,
            window=window,
            output_reserve=4096,
            counter=counter,
            top_k=int(config.get("top_k", 3)),
            threshold=threshold,
            include_seed=True,
            use_utility=False,
        )
        select_ms = (time.perf_counter() - started_select) * 1000
        lexical_candidate_map = _candidate_map(lexical_candidates)
        lexical_scored = {
            expected
            for expected in query.expected_ids
            if _score_stage(
                query,
                expected,
                lexical_candidate_map,
                recall_result.focus,
                threshold,
                attribute,
                recall_result.seed.id,
                blocked_policy,
                attribute_targets(query.question),
                {message.id: message for message in shortlist.messages},
            )
        }
        lexical_final = {
            expected for expected in query.expected_ids
            if _id_in_messages(expected, lexical.messages)
        }
        lexical_missing = None
        if query.expected_ids:
            if not set(query.expected_ids) <= shortlist.ids:
                lexical_missing = "candidates"
            elif not set(query.expected_ids) <= lexical_scored:
                lexical_missing = "scoring"
            elif not set(query.expected_ids) <= lexical_final:
                lexical_missing = "selection"
        outcomes.append(
            QueryOutcome(
                query=query.id,
                condition="lexical",
                candidate_ids=frozenset(
                    expected for expected in query.expected_ids if expected in shortlist.ids
                ),
                scored_ids=frozenset(lexical_scored),
                final_ids=frozenset(lexical_final),
                missing_stage=lexical_missing,
                sensitivity_excluded=query.sensitivity_expected and "t2" in blocked_policy,
                no_answer_empty=(
                    not lexical.diagnostics.selected_ids
                    and not lexical.diagnostics.seed_included
                    if query.no_answer
                    else None
                ),
                search_ms=shortlist.elapsed_ms,
                select_ms=select_ms,
                points=len(shortlist.messages),
                tokens=lexical.diagnostics.used_tokens,
            )
        )

    return {
        "scale": scale,
        "messages": len(messages),
        "build_ms": round(generated * 1000, 2),
        "index_ms": round(index_time * 1000, 2),
        "candidate_k": candidate_k,
        "outcomes": [outcome.as_dict() for outcome in outcomes],
        "summary": _summarize(outcomes),
    }


def _summarize(outcomes: Sequence[QueryOutcome]) -> dict:
    summary: dict[str, Any] = {}
    for condition in ("sinapse", "sinapse_curioso", "lexical"):
        selected = [outcome for outcome in outcomes if outcome.condition == condition]
        # Denominador explícito: consultas com evidência esperada e sem
        # exclusão intencional de sensibilidade.
        denominador = [
            outcome
            for outcome in selected
            if outcome.query
            not in {"sensibilidade", "preco_ausente", "quantidade"}
        ]
        candidate_all = sum(
            1 for outcome in denominador if outcome.missing_stage != "candidates"
        )
        final_all = sum(
            1 for outcome in denominador if outcome.missing_stage is None
        )
        search = [outcome.search_ms for outcome in selected]
        selection = [outcome.select_ms for outcome in selected]
        no_answer = [
            outcome for outcome in selected if outcome.no_answer_empty is not None
        ]
        summary[condition] = {
            "denominator": len(denominador),
            "candidate_all": candidate_all,
            "candidate_all_rate": round(candidate_all / len(denominador), 4)
            if denominador
            else 0.0,
            "final_all": final_all,
            "final_all_rate": round(final_all / len(denominador), 4)
            if denominador
            else 0.0,
            "search_median_ms": _percentile(search, 0.5),
            "search_p95_ms": _percentile(search, 0.95),
            "select_median_ms": _percentile(selection, 0.5),
            "select_p95_ms": _percentile(selection, 0.95),
            "avg_points": round(
                sum(outcome.points for outcome in selected) / len(selected), 1
            ),
            "avg_tokens": round(
                sum(outcome.tokens for outcome in selected) / len(selected), 1
            ),
            "no_answer_empty": sum(
                1 for outcome in no_answer if outcome.no_answer_empty
            ),
            "no_answer_total": len(no_answer),
            "sensitivity_excluded": sum(
                1 for outcome in selected if outcome.sensitivity_excluded
            ),
        }

    expected_by_query = {query.id: set(query.expected_ids) for query in QUERIES}
    off = {
        outcome.query: outcome
        for outcome in outcomes
        if outcome.condition == "sinapse"
    }
    on = {
        outcome.query: outcome
        for outcome in outcomes
        if outcome.condition == "sinapse_curioso"
    }
    additional = 0
    worsened: list[str] = []
    extra_candidates = 0
    for query_id, on_outcome in on.items():
        off_outcome = off[query_id]
        expected = expected_by_query[query_id]
        on_gain = expected & set(on_outcome.final_ids) - set(off_outcome.final_ids)
        if on_gain:
            additional += len(on_gain)
        if expected and expected <= set(off_outcome.final_ids) and not expected <= set(on_outcome.final_ids):
            worsened.append(query_id)
        if (
            off_outcome.no_answer_empty is True
            and on_outcome.no_answer_empty is False
        ):
            worsened.append(query_id)
        extra_candidates += on_outcome.extra_candidates
    summary["curiosity_comparison"] = {
        "additional_correct_evidence": additional,
        "worsened_cases": sorted(set(worsened)),
        "extra_candidates_evaluated": extra_candidates,
        "select_ms_delta_median": round(
            summary["sinapse_curioso"]["select_median_ms"]
            - summary["sinapse"]["select_median_ms"],
            2,
        ),
        "tokens_delta_avg": round(
            summary["sinapse_curioso"]["avg_tokens"]
            - summary["sinapse"]["avg_tokens"],
            1,
        ),
    }
    return summary
