"""Agente de memória com quadro branco e atenção temática por conversa.

Fluxo determinístico antes de cada resposta:

1. interpreta o turno atual com histórico recente e temas anteriores;
2. produz um relatório de temas, pesos, sinais e dimensão pedida;
3. usa os pesos para enviesar a semente e reordenar vizinhanças;
4. reavalia os itens que já estavam no quadro e busca novas evidências;
5. aplica o orçamento no formato de mensagens de chat;
6. registra entradas, permanências e saídas com scores e motivos.

O quadro guarda apenas IDs e metadados; as mensagens originais continuam
intactas no arquivo de memória. Sair do quadro não apaga nada.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .attributes import (
    ATTRIBUTE_PRICE,
    attribute_targets,
    requested_attribute,
    supporting_ids as attribute_supporting_ids,
)
from .budget import RecentTurn, SimulatedCounter
from .chat import (
    ChatBudgetResult,
    ChatTokenCounter,
    EstimatedChatCounter,
    select_chat_context,
)
from .config import load_config
from .deepseek import DeepSeekSettings, build_request_body
from .memory import (
    association as association_for,
    evidence_vector,
    recall,
    weighted_distance,
)
from .models import Candidate, Message, RecallResult
from .sensitivity import (
    ACTION_ALLOW,
    ACTION_DELETE,
    ACTION_RESTRICT,
    SOURCE_DECLARED,
    DerivedSummary,
    PreferenceEvent,
    SensitivityPolicy,
    SensitivityReport,
    SummaryDecision,
)
from .themes import (
    ThemeReport,
    analyze_turn,
    theme_affinity,
    themes_for_message,
)
from .text import infer_dimension, normalize_text, tokenize

ASSUNTO = "assunto"
EVIDENCIA = "evidência"
THEME_FLOOR = 0.6


def _default_window(config: Mapping[str, Any]) -> int:
    deepseek = config.get("deepseek") or {}
    return int(deepseek.get("context_window", config.get("window", 2048)))


def _default_reserve(config: Mapping[str, Any]) -> int:
    deepseek = config.get("deepseek") or {}
    return int(deepseek.get("output_reserve", config.get("output_reserve", 512)))


@dataclass(frozen=True)
class WhiteboardItem:
    message_id: str
    role: str
    relation: float
    distance: float
    association: float
    attention: float
    base_attention: float
    evidence_focus: float | None
    themes: tuple[str, ...]
    entered_turn: int
    updated_turn: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "relation": self.relation,
            "distance": self.distance,
            "association": self.association,
            "attention": self.attention,
            "base_attention": self.base_attention,
            "evidence_focus": self.evidence_focus,
            "themes": list(self.themes),
            "entered_turn": self.entered_turn,
            "updated_turn": self.updated_turn,
            "reason": self.reason,
        }


@dataclass
class ConversationWhiteboard:
    conversation_id: str
    turn: int = 0
    items: dict[str, WhiteboardItem] = field(default_factory=dict)
    theme_weights: dict[str, float] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)

    def ids(self) -> tuple[str, ...]:
        return tuple(self.items)

    def ordered(self) -> tuple[WhiteboardItem, ...]:
        return tuple(
            sorted(
                self.items.values(),
                key=lambda item: (0 if item.role == ASSUNTO else 1, -item.attention, item.message_id),
            )
        )

    def as_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "turn": self.turn,
            "theme_weights": dict(self.theme_weights),
            "items": [item.as_dict() for item in self.ordered()],
        }


@dataclass(frozen=True)
class BoardDelta:
    entered: tuple[str, ...]
    stayed: tuple[str, ...]
    left: tuple[str, ...]
    left_reasons: Mapping[str, str]

    def as_dict(self) -> dict:
        return {
            "entered": list(self.entered),
            "stayed": list(self.stayed),
            "left": list(self.left),
            "left_reasons": dict(self.left_reasons),
        }


@dataclass(frozen=True)
class BoardDecision:
    message_id: str
    action: str
    role: str
    distance: float
    association: float
    attention: float
    base_attention: float
    evidence_focus: float | None
    themes: tuple[str, ...]
    reason: str
    detail: str | None = None
    sent: bool = False
    budget_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "role": self.role,
            "distance": self.distance,
            "association": self.association,
            "attention": self.attention,
            "base_attention": self.base_attention,
            "evidence_focus": self.evidence_focus,
            "themes": list(self.themes),
            "reason": self.reason,
            "detail": self.detail,
            "sent": self.sent,
            "budget_reason": self.budget_reason,
        }


@dataclass(frozen=True)
class CuriosityReport:
    enabled: bool
    triggered: bool
    signal: str | None
    reason: str | None
    evaluated_ids: tuple[str, ...]
    new_in_context: tuple[str, ...]
    doubt: bool
    doubt_reason: str | None
    elapsed_ms: float
    max_extra: int

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "triggered": self.triggered,
            "signal": self.signal,
            "reason": self.reason,
            "evaluated_ids": list(self.evaluated_ids),
            "new_in_context": list(self.new_in_context),
            "doubt": self.doubt,
            "doubt_reason": self.doubt_reason,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "max_extra": self.max_extra,
        }


@dataclass(frozen=True)
class AgentTurn:
    conversation_id: str
    turn_index: int
    query: str
    focus: str
    focus_rule: str
    seed: Message
    themes: ThemeReport
    sensitivity: SensitivityReport
    summaries: tuple[DerivedSummary, ...]
    curiosity: CuriosityReport
    board: tuple[WhiteboardItem, ...]
    delta: BoardDelta
    recall: RecallResult
    budget: ChatBudgetResult
    decisions: tuple[BoardDecision, ...]
    messages: tuple[dict[str, str], ...]
    store_size: int
    store_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        blocked = set(self.sensitivity.blocked_ids)
        recall_payload = self.recall.as_dict()
        for candidate in recall_payload["candidates"]:
            if candidate["id"] in blocked:
                candidate["text"] = "[restrito]"
        if recall_payload["seed"]["id"] in blocked:
            recall_payload["seed"]["text"] = "[restrito]"
        budget_payload = self.budget.diagnostics.as_dict()
        for decision in budget_payload["decisions"]:
            if decision["id"] in blocked:
                decision["text"] = "[restrito]"
        return {
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "query": self.query,
            "focus": self.focus,
            "focus_rule": self.focus_rule,
            "seed": {"id": self.seed.id, "restricted": self.seed.id in blocked},
            "themes": self.themes.as_dict(),
            "sensitivity": self.sensitivity.as_dict(),
            "curiosity": self.curiosity.as_dict(),
            "summaries": [summary.id for summary in self.summaries],
            "board": [item.as_dict() for item in self.board],
            "delta": self.delta.as_dict(),
            "recall": recall_payload,
            "budget": budget_payload,
            "decisions": [decision.as_dict() for decision in self.decisions],
            "messages": [dict(message) for message in self.messages],
            "store_size": self.store_size,
            "store_ids": list(self.store_ids),
        }


def _candidate_reason(
    candidate: Candidate, focus: str, threshold: float
) -> tuple[str, str | None]:
    if not candidate.within_radius:
        return "outside_radius", f"distância {candidate.distance:.3f} > raio"
    if candidate.evidence.get(focus) is None:
        return "no_focus_evidence", f"sem informação própria em {focus}"
    if candidate.attention <= threshold:
        return "below_threshold", (
            f"atenção {candidate.attention:.4f} <= limiar {threshold:.4f}"
        )
    return "board_limit", "elegível, mas fora do limite do quadro"


@dataclass(frozen=True)
class _Evaluated:
    candidate: Candidate
    base_attention: float
    affinity: float
    themes: tuple[str, ...]


class MemoryAgent:
    """Um quadro branco por conversa, com o mesmo núcleo de evocação."""

    def __init__(
        self,
        messages: Sequence[Message],
        config: Mapping[str, Any] | None = None,
        counter: ChatTokenCounter | None = None,
        board_limit: int = 5,
        curiosity: bool | None = None,
    ) -> None:
        self.messages = list(messages)
        self.by_id = {message.id: message for message in self.messages}
        self.config = dict(config or load_config())
        self.counter = counter or EstimatedChatCounter(SimulatedCounter())
        self.board_limit = int(board_limit)
        curiosity_config = self.config.get("curiosity", {}) or {}
        self.curiosity_enabled = bool(
            curiosity
            if curiosity is not None
            else curiosity_config.get("enabled", False)
        )
        self.max_extra_candidates = int(
            curiosity_config.get("max_extra_candidates", 6)
        )
        self.boards: dict[str, ConversationWhiteboard] = {}
        self.sensitivity = SensitivityPolicy()
        self.summaries: dict[str, list[DerivedSummary]] = {}
        self._summary_counter = 0

    def add_summary(
        self,
        conversation_id: str,
        text: str,
        source_ids: Sequence[str],
        summary_id: str | None = None,
    ) -> DerivedSummary:
        self._summary_counter += 1
        summary = DerivedSummary(
            id=summary_id or f"sum:{self._summary_counter}",
            conversation_id=conversation_id,
            text=text,
            source_ids=tuple(source_ids),
        )
        self.summaries.setdefault(conversation_id, []).append(summary)
        return summary

    def declare_preference(
        self, conversation_id: str, text: str, turn: int | None = None
    ) -> tuple[PreferenceEvent, ...]:
        board = self.board_for(conversation_id)
        return self.sensitivity.declare(
            conversation_id, text, board.turn + 1 if turn is None else turn
        )

    def infer_preference(
        self, conversation_id: str, text: str, turn: int | None = None
    ) -> tuple[PreferenceEvent, ...]:
        board = self.board_for(conversation_id)
        return self.sensitivity.infer(
            conversation_id, text, board.turn + 1 if turn is None else turn
        )

    def board_for(self, conversation_id: str) -> ConversationWhiteboard:
        if conversation_id not in self.boards:
            self.boards[conversation_id] = ConversationWhiteboard(conversation_id)
        return self.boards[conversation_id]

    @property
    def store_size(self) -> int:
        return len(self.messages)

    @property
    def store_ids(self) -> tuple[str, ...]:
        return tuple(message.id for message in self.messages)

    def message(self, message_id: str) -> Message:
        return self.by_id[message_id]

    def _evaluate(
        self,
        candidate: Candidate,
        weights: Mapping[str, float],
    ) -> _Evaluated:
        affinity = theme_affinity(candidate.message, weights, self.by_id)
        effective = candidate.attention * (THEME_FLOOR + (1.0 - THEME_FLOOR) * affinity)
        themed = replace(candidate, attention=effective)
        return _Evaluated(
            candidate=themed,
            base_attention=candidate.attention,
            affinity=affinity,
            themes=themes_for_message(candidate.message, weights, self.by_id),
        )

    def _curiosity_round(
        self,
        query: str,
        turns: Sequence[RecentTurn],
        theme_report: ThemeReport,
        recall_result,
        seed: Message,
        evaluated: Sequence[_Evaluated],
        conversation_messages: Sequence[Message],
        blocked: set[str],
        threshold: float,
    ) -> tuple[CuriosityReport, list[_Evaluated]]:
        started = time.perf_counter()
        if not self.curiosity_enabled:
            return (
                CuriosityReport(
                    enabled=False,
                    triggered=False,
                    signal=None,
                    reason=None,
                    evaluated_ids=(),
                    new_in_context=(),
                    doubt=False,
                    doubt_reason=None,
                    elapsed_ms=0.0,
                    max_extra=self.max_extra_candidates,
                ),
                [],
            )

        attribute = requested_attribute(query)
        targets = attribute_targets(query)
        supported = attribute_supporting_ids(
            conversation_messages, attribute, "c1", targets, self.by_id
        ) if attribute else frozenset()
        initial_evidence = [
            item
            for item in evaluated
            if item.candidate.evidence.get(recall_result.focus) is not None
            and item.candidate.attention > threshold
        ]
        texts = [query] + [turn.text for turn in turns] + [
            item.candidate.message.text for item in evaluated[:3]
        ]
        entities = {
            match.lower()
            for match in re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(texts))
        }
        entities -= {
            "falamos",
            "escolhi",
            "testei",
            "passei",
            "troquei",
            "quanto",
            "como",
            "porque",
            "voltando",
            "preparei",
        }
        entities |= set(theme_report.weights())
        normalized = normalize_text(query)
        pronoun = bool(
            re.search(r"\b(?:ele|ela|eles|elas|isso|isto|aquilo|esse|essa)\b", normalized)
        )
        short = len(tokenize(query)) <= 3
        correction_terms = (
            "troquei",
            "deixei",
            "corrigi",
            "na verdade",
            "agora uso",
            "mudei",
        )
        correction_messages = [
            message
            for message in conversation_messages
            if any(term in normalize_text(message.text) for term in correction_terms)
        ]
        initial_ids = {item.candidate.message.id for item in evaluated}

        signal: str | None = None
        reason: str | None = None
        extra_messages: list[Message] = []

        if pronoun and short and len(entities) >= 2:
            signal = "ambiguous_reference"
            reason = (
                "referência curta com múltiplos temas plausíveis: "
                + ", ".join(sorted(entities)[:5])
            )
            for message in conversation_messages:
                if message.id in initial_ids or message.id in blocked:
                    continue
                tokens = set(tokenize(message.text))
                if tokens & entities:
                    extra_messages.append(message)
        elif (attribute and not supported) or not initial_evidence:
            signal = "missing_evidence"
            reason = "informação pedida ainda sem evidência; seguir vínculos de resposta"
            link_ids: set[str] = set()
            for item in [seed, *[entry.candidate.message for entry in evaluated[:3]]]:
                if item.reply_to:
                    link_ids.add(item.reply_to)
                link_ids.update(
                    message.id
                    for message in conversation_messages
                    if message.reply_to == item.id
                )
            for message in conversation_messages:
                if message.id in link_ids and message.id not in blocked:
                    extra_messages.append(message)
        elif correction_messages and any(
            term in normalized for term in correction_terms
        ):
            signal = "decision_conflict"
            reason = "há revisão/correção sobre a mesma decisão"
            for message in correction_messages:
                if message.id not in blocked:
                    extra_messages.append(message)

        if not signal:
            return (
                CuriosityReport(
                    enabled=True,
                    triggered=False,
                    signal=None,
                    reason=None,
                    evaluated_ids=(),
                    new_in_context=(),
                    doubt=False,
                    doubt_reason=None,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    max_extra=self.max_extra_candidates,
                ),
                [],
            )

        extras: list[_Evaluated] = []
        seen = set(initial_ids)
        for message in extra_messages:
            if message.id in seen or message.id in blocked:
                continue
            seen.add(message.id)
            distance = weighted_distance(message, seed, recall_result.weights)
            association_value, content, link, source = association_for(
                message,
                seed,
                recall_result.anchor_tokens,
                float(self.config.get("link_association_floor", 0.9)),
            )
            evidence = evidence_vector(message, seed, recall_result.focus)
            attention = association_value * (evidence.get(recall_result.focus) or 0.0)
            candidate = Candidate(
                message=message,
                distance=distance,
                association=association_value,
                association_content=content,
                association_link=link,
                association_source=source,
                evidence=evidence,
                attention=attention,
                within_radius=distance <= float(recall_result.radius),
            )
            extras.append(
                _Evaluated(
                    candidate=candidate,
                    base_attention=attention,
                    affinity=theme_affinity(message, theme_report.weights(), self.by_id),
                    themes=themes_for_message(message, theme_report.weights(), self.by_id),
                )
            )
            if len(extras) >= self.max_extra_candidates:
                break

        doubt = not extras
        doubt_reason = (
            "nenhum candidato adicional encontrado para o sinal"
            if not extras
            else None
        )
        return (
            CuriosityReport(
                enabled=True,
                triggered=True,
                signal=signal,
                reason=reason,
                evaluated_ids=tuple(extra.candidate.message.id for extra in extras),
                new_in_context=(),
                doubt=doubt,
                doubt_reason=doubt_reason,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                max_extra=self.max_extra_candidates,
            ),
            extras,
        )

    def before_response(
        self,
        conversation_id: str,
        query: str,
        recent_turns: Sequence[RecentTurn | str] = (),
        *,
        window: int | None = None,
        output_reserve: int | None = None,
        seed_id: str | None = None,
        focus: str | None = None,
        radius: float | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        include_seed: bool = True,
        base_message_ids: Sequence[str] = (),
    ) -> AgentTurn:
        turns = tuple(
            turn
            if isinstance(turn, RecentTurn)
            else RecentTurn(
                text=str(turn["text"]),
                id=turn.get("id"),
                author=str(turn.get("author", "user")),
                position=turn.get("position"),
            )
            if isinstance(turn, Mapping)
            else RecentTurn(text=str(turn))
            for turn in recent_turns
        )
        board = self.board_for(conversation_id)
        board.turn += 1
        turn_index = board.turn
        previous = dict(board.items)
        previous_ids = tuple(previous)

        previous_dominant = (
            max(board.theme_weights, key=board.theme_weights.get)
            if board.theme_weights
            else None
        )
        events = tuple(
            self.sensitivity.declare(conversation_id, query, turn_index)
        ) + tuple(
            self.sensitivity.infer(
                conversation_id, query, turn_index, previous_dominant
            )
        )
        raw_sensitivity = self.sensitivity.evaluate(
            conversation_id, query, self.messages, turn_index
        )
        blocked = set(raw_sensitivity.blocked_ids)
        deleted = set(raw_sensitivity.deleted_ids)
        if deleted:
            self.messages = [
                message for message in self.messages if message.id not in deleted
            ]
        conversation_messages = [
            message
            for message in self.messages
            if message.conversation_id == conversation_id and message.id not in blocked
        ]
        removed_from_board = tuple(
            message_id for message_id in previous_ids if message_id in blocked
        )
        active_previous_ids = [
            message_id for message_id in previous_ids if message_id not in blocked
        ]
        board_messages = [
            self.by_id[message_id]
            for message_id in active_previous_ids
            if message_id in self.by_id
        ]
        board_context = [message.text for message in board_messages]

        focus_override = focus
        focus_rule_override: str | None = None
        if focus is None and raw_sensitivity.released_ids:
            inferred, rule = infer_dimension(query)
            if not rule.startswith("fallback"):
                focus_override = inferred
            else:
                priority = (
                    "por_que",
                    "como",
                    "para_que",
                    "o_que",
                    "quem",
                    "onde",
                    "quando",
                )
                released_dimensions = {
                    dimension
                    for message_id in raw_sensitivity.released_ids
                    for dimension in self.by_id[message_id].coordinates
                    if self.by_id[message_id].values(dimension)
                }
                for dimension in priority:
                    if dimension in released_dimensions:
                        focus_override = dimension
                        focus_rule_override = "dimensão do detalhe liberado (pedido claro)"
                        break

        theme_report = analyze_turn(
            query,
            [turn.text for turn in turns],
            board.theme_weights,
            board_context,
        )
        weights = theme_report.weights()
        explicit = bool(theme_report.explicit_themes)
        current_tokens = tokenize(query)
        seed_bias: dict[str, float] = {}
        for message in conversation_messages:
            bias = 1.0 + 0.5 * theme_affinity(message, weights, self.by_id)
            if not explicit and not theme_report.dimension_explicit and current_tokens:
                # Referência curta sem dimensão explícita ("E aquilo que eu
                # preparei?"): os termos do turno atual desempatam o assunto.
                direct = tokenize(message.text)
                bias += 0.8 * len(direct & current_tokens) / len(current_tokens)
            seed_bias[message.id] = bias
        # Menção explícita no turno manda: histórico e quadro anterior não
        # sequestram o assunto novo. Sem menção explícita, ambos ajudam a
        # resolver referências curtas ("E como eu testei?").
        anchor_context = () if explicit else board_context

        recall_result = recall(
            self.messages,
            query,
            [turn.text for turn in turns],
            self.config,
            seed_id=seed_id,
            focus=focus_override,
            radius=radius,
            conversation_id=conversation_id,
            board_context=anchor_context,
            seed_bias=seed_bias,
            anchor_turns=() if explicit else None,
            excluded_ids=tuple(blocked),
        )
        seed = recall_result.seed
        seed_allowed = seed.id not in blocked
        effective_threshold = float(
            threshold
            if threshold is not None
            else self.config.get("attention_threshold", 0.05)
        )

        evaluated = [self._evaluate(candidate, weights) for candidate in recall_result.candidates]
        by_evaluated = {item.candidate.message.id: item for item in evaluated}

        attribute_suppressed = False
        attribute = requested_attribute(query)
        if attribute == ATTRIBUTE_PRICE:
            supported = attribute_supporting_ids(
                conversation_messages,
                attribute,
                conversation_id,
                attribute_targets(query),
                self.by_id,
            )
            if not supported:
                attribute_suppressed = True
                evaluated = []
            else:
                augmented: list[_Evaluated] = []
                for item in evaluated:
                    if item.candidate.message.id not in supported:
                        continue
                    candidate = replace(
                        item.candidate,
                        evidence={
                            **item.candidate.evidence,
                            recall_result.focus: 1.0,
                        },
                        attention=max(
                            item.candidate.attention, item.candidate.association
                        ),
                    )
                    augmented.append(
                        _Evaluated(
                            candidate=candidate,
                            base_attention=item.base_attention,
                            affinity=item.affinity,
                            themes=item.themes,
                        )
                    )
                evaluated = augmented
        if attribute_suppressed:
            seed_allowed = False

        curiosity, extra_evaluated = self._curiosity_round(
            query,
            turns,
            theme_report,
            recall_result,
            seed,
            evaluated,
            conversation_messages,
            blocked,
            effective_threshold,
        )
        if extra_evaluated:
            existing = {item.candidate.message.id for item in evaluated}
            evaluated = list(evaluated) + [
                item
                for item in extra_evaluated
                if item.candidate.message.id not in existing
            ]
            by_evaluated = {
                item.candidate.message.id: item for item in evaluated
            }

        summary_decisions: list[SummaryDecision] = []
        allowed_summaries: list[DerivedSummary] = []
        for summary in self.summaries.get(conversation_id, []):
            is_blocked, summary_reason = self.sensitivity.summary_is_blocked(
                conversation_id, summary, query
            )
            summary_decisions.append(
                SummaryDecision(
                    summary_id=summary.id,
                    allowed=not is_blocked,
                    reason=summary_reason,
                    source="preferencia",
                )
            )
            if not is_blocked:
                allowed_summaries.append(summary)
        summary_text = "\n\n".join(summary.text for summary in allowed_summaries)

        budget_turns: list[RecentTurn] = []
        omitted_history: list[str] = []
        for index, turn in enumerate(turns):
            if turn.id and turn.id in blocked:
                omitted_history.append(turn.id)
                continue
            text_blocked, _ = self.sensitivity.text_is_blocked(
                conversation_id, turn.text, query
            )
            if text_blocked:
                omitted_history.append(turn.id or f"hist:{index}")
                continue
            budget_turns.append(turn)

        # Reavalia o que já estava no quadro e busca novas evidências elegíveis.
        eligible = [
            item
            for item in evaluated
            if item.candidate.within_radius
            and item.candidate.evidence.get(recall_result.focus) is not None
            and item.candidate.attention > effective_threshold
        ]
        eligible.sort(
            key=lambda item: (
                -item.candidate.attention,
                item.candidate.distance,
                item.candidate.message.position,
                item.candidate.message.id,
            )
        )
        board_capacity = 1 if theme_report.ambiguous else self.board_limit
        chosen = eligible[:board_capacity]

        seed_themes = themes_for_message(seed, weights, self.by_id)
        new_items: dict[str, WhiteboardItem] = {}
        if seed_allowed:
            new_items[seed.id] = WhiteboardItem(
                message_id=seed.id,
                role=ASSUNTO,
                relation=1.0,
                distance=0.0,
                association=1.0,
                attention=1.0,
                base_attention=1.0,
                evidence_focus=None,
                themes=seed_themes,
                entered_turn=previous[seed.id].entered_turn
                if seed.id in previous
                else turn_index,
                updated_turn=turn_index,
                reason="semente do turno",
            )
        for rank, item in enumerate(chosen, start=1):
            candidate = item.candidate
            message_id = candidate.message.id
            was_on_board = message_id in previous
            reason = "permaneceu no quadro" if was_on_board else "nova evidência"
            if theme_report.ambiguous and rank == 1:
                reason = "evidência única em turno ambíguo"
            new_items[message_id] = WhiteboardItem(
                message_id=message_id,
                role=EVIDENCIA,
                relation=candidate.association,
                distance=candidate.distance,
                association=candidate.association,
                attention=candidate.attention,
                base_attention=item.base_attention,
                evidence_focus=candidate.evidence.get(recall_result.focus),
                themes=item.themes,
                entered_turn=previous[message_id].entered_turn
                if was_on_board
                else turn_index,
                updated_turn=turn_index,
                reason=reason,
            )

        new_ids = tuple(new_items)
        entered = tuple(
            sorted(
                (message_id for message_id in new_ids if message_id not in previous),
                key=lambda message_id: (self.by_id[message_id].position, message_id),
            )
        )
        stayed = tuple(
            sorted(
                (message_id for message_id in new_ids if message_id in previous),
                key=lambda message_id: (self.by_id[message_id].position, message_id),
            )
        )
        left = tuple(
            sorted(
                (message_id for message_id in previous_ids if message_id not in new_items),
                key=lambda message_id: (self.by_id[message_id].position, message_id),
            )
        )
        left_reasons: dict[str, str] = {}
        for message_id in left:
            if attribute_suppressed:
                left_reasons[message_id] = "no_pertinent_evidence"
                continue
            if message_id in blocked:
                left_reasons[message_id] = "sensitivity_blocked"
                continue
            item = by_evaluated.get(message_id)
            if item is None:
                left_reasons[message_id] = "fora_da_conversa"
                continue
            reason, _ = _candidate_reason(
                item.candidate, recall_result.focus, effective_threshold
            )
            left_reasons[message_id] = reason
        delta = BoardDelta(
            entered=entered,
            stayed=stayed,
            left=left,
            left_reasons=left_reasons,
        )

        board.items = new_items
        board.theme_weights = weights
        board.log.append(
            {
                "turn": turn_index,
                "query": query,
                "seed": seed.id,
                "focus": recall_result.focus,
                "themes": dict(weights),
                "ambiguous": theme_report.ambiguous,
                "board_ids": list(new_ids),
                "entered": list(entered),
                "stayed": list(stayed),
                "left": list(left),
                "left_reasons": dict(left_reasons),
            }
        )

        instructions = str(self.config.get("instructions", ""))
        if summary_text:
            instructions += "\n\n=== RESUMO (Sinapse) ===\n" + summary_text
        budget = select_chat_context(
            instructions=instructions,
            recent_turns=tuple(budget_turns),
            current_turn=query,
            seed=seed,
            focus=recall_result.focus,
            candidates=[item.candidate for item in chosen],
            window=int(window if window is not None else _default_window(self.config)),
            output_reserve=int(
                output_reserve
                if output_reserve is not None
                else _default_reserve(self.config)
            ),
            counter=self.counter,
            top_k=int(top_k if top_k is not None else self.config.get("top_k", 3)),
            threshold=effective_threshold,
            include_seed=include_seed and seed_allowed,
            explicit_message_ids=base_message_ids,
            min_gain=float(self.config.get("min_marginal_gain", 0.05)),
            min_novelty=float(self.config.get("min_marginal_novelty", 0.55)),
        )
        selected = set(budget.diagnostics.selected_ids)
        budget_reasons = {
            decision.id: decision.reason for decision in budget.diagnostics.decisions
        }
        new_in_context = tuple(
            message_id
            for message_id in curiosity.evaluated_ids
            if message_id in selected
        )
        if curiosity.triggered:
            if new_in_context:
                curiosity = replace(
                    curiosity,
                    new_in_context=new_in_context,
                    doubt=False,
                    doubt_reason=None,
                )
            else:
                curiosity = replace(
                    curiosity,
                    new_in_context=(),
                    doubt=True,
                    doubt_reason="evidência adicional não entrou no contexto final",
                )

        decisions: list[BoardDecision] = []
        for item in board.ordered():
            action = "entered" if item.message_id in entered else "stayed"
            if item.role == ASSUNTO:
                decisions.append(
                    BoardDecision(
                        message_id=item.message_id,
                        action=action,
                        role=item.role,
                        distance=item.distance,
                        association=item.association,
                        attention=item.attention,
                        base_attention=item.base_attention,
                        evidence_focus=item.evidence_focus,
                        themes=item.themes,
                        reason=item.reason,
                        detail="referência de assunto; entra se o orçamento permitir",
                        sent=budget.diagnostics.seed_included
                        or budget.diagnostics.seed_in_base,
                        budget_reason=budget.diagnostics.seed_source,
                    )
                )
                continue
            decisions.append(
                BoardDecision(
                    message_id=item.message_id,
                    action=action,
                    role=item.role,
                    distance=item.distance,
                    association=item.association,
                    attention=item.attention,
                    base_attention=item.base_attention,
                    evidence_focus=item.evidence_focus,
                    themes=item.themes,
                    reason=item.reason,
                    sent=item.message_id in selected,
                    budget_reason=budget_reasons.get(item.message_id),
                )
            )
        previous_order = sorted(
            previous_ids,
            key=lambda message_id: (self.by_id[message_id].position, message_id),
        )
        for message_id in previous_order:
            if message_id in new_items:
                continue
            if message_id in blocked:
                decisions.append(
                    BoardDecision(
                        message_id=message_id,
                        action="left",
                        role=previous[message_id].role,
                        distance=previous[message_id].distance,
                        association=previous[message_id].association,
                        attention=previous[message_id].attention,
                        base_attention=previous[message_id].base_attention,
                        evidence_focus=previous[message_id].evidence_focus,
                        themes=previous[message_id].themes,
                        reason="sensitivity_blocked",
                        detail="restrição de sensibilidade aplicada antes da atenção",
                        sent=False,
                        budget_reason=None,
                    )
                )
                continue
            item = by_evaluated.get(message_id)
            if item is None:
                reason, detail = "fora_da_conversa", None
            else:
                reason, detail = _candidate_reason(
                    item.candidate, recall_result.focus, effective_threshold
                )
            decisions.append(
                BoardDecision(
                    message_id=message_id,
                    action="left",
                    role=previous[message_id].role,
                    distance=item.candidate.distance if item else 1.0,
                    association=item.candidate.association if item else 0.0,
                    attention=item.candidate.attention if item else 0.0,
                    base_attention=item.base_attention if item else 0.0,
                    evidence_focus=item.candidate.evidence.get(recall_result.focus)
                    if item
                    else None,
                    themes=item.themes if item else (),
                    reason=reason,
                    detail=detail,
                    sent=False,
                    budget_reason=None,
                )
            )

        final_sensitivity = self.sensitivity.with_context(
            raw_sensitivity,
            removed_from_board=removed_from_board,
            omitted_history=omitted_history,
            summaries=summary_decisions,
            events=events,
        )
        return AgentTurn(
            conversation_id=conversation_id,
            turn_index=turn_index,
            query=query,
            focus=recall_result.focus,
            focus_rule=recall_result.focus_rule,
            seed=seed,
            themes=theme_report,
            sensitivity=final_sensitivity,
            summaries=tuple(allowed_summaries),
            curiosity=curiosity,
            board=board.ordered(),
            delta=delta,
            recall=recall_result,
            budget=budget,
            decisions=tuple(decisions),
            messages=budget.messages,
            store_size=len(self.messages),
            store_ids=tuple(message.id for message in self.messages),
        )


def send_agent_turn(
    turn: AgentTurn,
    client,
    settings: DeepSeekSettings,
) -> tuple[dict[str, Any], Any]:
    """Monta o corpo final a partir do quadro e envia pelo transporte dado."""
    body = build_request_body(turn.messages, settings)
    response = client.complete(body, timeout=settings.timeout)
    return body, response
