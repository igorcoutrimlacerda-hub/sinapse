"""Sensibilidade: preferências declaradas, inferências e restrição de evocação.

Duas camadas separadas:

- ``Preference`` com ``source="declarada"``: o usuário declarou explicitamente
  uma restrição, uma exclusão ou uma permissão. Tem prioridade e pode ser
  atualizada.
- ``Preference`` com ``source="inferencia"``: o avaliador temático inferiu
  desconforto por sinais indiretos. Bloqueia evocação espontânea, mas não
  vence uma preferência explícita.

Escopos: assunto, mensagem ou detalhe. A restrição é aplicada **antes** da
ordenação por atenção e do orçamento. Um pedido claro e específico no turno
atual libera apenas o alvo pedido, só naquele turno. Exclusão é permanente e
distinta de restrição.

Diagnóstico: apenas IDs, ações, escopos e motivos. O alvo restrito nunca é
reproduzido nos relatórios.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import Message
from .text import normalize_text, tokenize, tokenize_values
from .themes import THEME_LEXICONS

SCOPE_SUBJECT = "assunto"
SCOPE_MESSAGE = "mensagem"
SCOPE_DETAIL = "detalhe"

ACTION_RESTRICT = "restringir"
ACTION_DELETE = "excluir"
ACTION_ALLOW = "permitir"

SOURCE_DECLARED = "declarada"
SOURCE_INFERENCE = "inferencia"

RELEASE_MARKER = re.compile(
    r"\b(?:lembra|lembrar|recorda|recordar|retomar|retome|retoma|voltar\s+a|"
    r"fale\s+sobre|falar\s+sobre|pode\s+falar|quero\s+falar|me\s+conta|"
    r"me\s+conte|conte|conta|mostrar|mostre|especificamente|detalhe|detalhes|"
    r"de\s+novo)\b"
)

_DELETE_PATTERNS = (
    r"\b(?:apague|apagar|delete|deletar|remova|remover|esqueca|esquecer)\s+"
    r"(?:a\s+)?(?:mensagem\s+)?(?P<target>[a-z0-9 ]{1,80})",
)
_ALLOW_PATTERNS = (
    r"\b(?:pode|quero|vamos)\s+(?:voltar\s+a\s+)?(?:falar|conversar)\s+"
    r"(?:sobre|de|do|da)\s+(?P<target>[a-z0-9 ]{1,80})",
    r"\b(?:libere|liberar|permita|permitir)\s+"
    r"(?:o\s+assunto\s+|a\s+mensagem\s+|o\s+detalhe\s+)?(?P<target>[a-z0-9 ]{1,80})",
    r"\bpode\s+lembrar\s+(?:de|do|da)\s+(?P<target>[a-z0-9 ]{1,80})",
)
_RESTRICT_PATTERNS = (
    r"\bnao\s+quero\s+(?:mais\s+)?(?:falar|conversar)\s+(?:sobre|de|do|da)\s+"
    r"(?P<target>[a-z0-9 ]{1,80})",
    r"\bnao\s+(?:fale|falar|mencione|mencionar|toque|tocar|traga|trazer)\s+"
    r"(?:mais\s+)?(?:sobre|em|de|do|da|no|na)?\s*(?P<target>[a-z0-9 ]{1,80})",
    r"\bprefiro\s+(?:que\s+voce\s+)?nao\s+(?:falar|mencionar|tocar)\s+"
    r"(?:sobre|em|de|do|da)?\s*(?P<target>[a-z0-9 ]{1,80})",
    r"\brestrinja\s+(?:(?:o\s+assunto|a\s+mensagem)\s+)?"
    r"(?P<target>[a-z0-9 ]{1,80})",
)
_INFERENCE_PATTERNS = (
    r"(?P<target>[a-z0-9 ]{0,60})\b(?:me\s+deixa|me\s+deixam)\s+desconfortavel",
    r"\b(?:me\s+sinto|fico)\s+mal\s+(?:com|falando\s+de|quando\s+se\s+fala\s+de)\s+"
    r"(?P<target>[a-z0-9 ]{1,80})",
    r"\b(?:e|e\s+um)\s+assunto\s+sensivel\b"
    r"(?:\s+(?:de|do|da|sobre)\s+(?P<target>[a-z0-9 ]{1,80}))?",
    r"\b(?:estou|ando)\s+evitando\s+(?P<target>[a-z0-9 ]{1,80})",
    r"\bnao\s+e\s+um\s+bom\s+momento\s+(?:para|de)\s+(?P<target>[a-z0-9 ]{1,80})",
)

_LEADING = re.compile(
    r"^(?:esse|essa|este|esta|aquele|aquela|o|a|os|as|um|uma|do|da|dos|das|de)\s+"
)
_TOPIC_LEADING = re.compile(
    r"^(?:assunto|detalhe|detalhes|mensagem)\s+(?:(?:de|do|da|dos|das|sobre)\s+)?"
)
_PREPOSITIONS = frozenset(
    {"de", "do", "da", "dos", "das", "no", "na", "em", "para", "com", "que", "e"}
)


def _clean_phrase(phrase: str) -> str:
    text = normalize_text(phrase)
    text = _LEADING.sub("", text)
    text = _TOPIC_LEADING.sub("", text)
    text = _LEADING.sub("", text)
    return text.strip()


def _head_phrase(phrase: str) -> str:
    words = _clean_phrase(phrase).split()
    head: list[str] = []
    for word in words:
        if word in _PREPOSITIONS:
            break
        head.append(word)
    return " ".join(head).strip()


def _message_tokens(message: Message) -> frozenset[str]:
    tokens: set[str] = set(tokenize(message.text))
    for dimension in message.coordinates:
        tokens.update(tokenize_values(message.values(dimension)))
    return frozenset(tokens)


def _classify(phrase: str) -> tuple[str, str] | None:
    cleaned = _clean_phrase(phrase)
    if not cleaned:
        return None
    message_ids = re.findall(r"\bm\d+\b", cleaned)
    if message_ids:
        return SCOPE_MESSAGE, message_ids[0]
    if re.search(r"\b(?:detalhe|detalhes)\b", normalize_text(phrase)):
        head = _head_phrase(cleaned) or cleaned
        tokens = tokenize(head)
        if not tokens:
            return None
        return SCOPE_DETAIL, " ".join(sorted(tokens))
    head = _head_phrase(cleaned) or cleaned
    tokens = tokenize(head)
    if not tokens:
        return None
    for theme, lexicon in THEME_LEXICONS.items():
        if head == theme or tokens & lexicon:
            return SCOPE_SUBJECT, theme
    return SCOPE_DETAIL, " ".join(sorted(tokens))


@dataclass(frozen=True)
class Preference:
    id: str
    conversation_id: str
    scope: str
    target: str
    action: str
    source: str
    confidence: float
    created_turn: int
    updated_turn: int


@dataclass(frozen=True)
class PreferenceEvent:
    preference_id: str
    operation: str
    action: str
    scope: str
    source: str
    turn: int

    def as_dict(self) -> dict:
        return {
            "preference_id": self.preference_id,
            "operation": self.operation,
            "action": self.action,
            "scope": self.scope,
            "source": self.source,
            "turn": self.turn,
        }


@dataclass(frozen=True)
class SensitivityDecision:
    message_id: str
    allowed: bool
    action: str
    scope: str
    source: str
    reason: str
    released: bool
    confidence: float

    def as_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "allowed": self.allowed,
            "action": self.action,
            "scope": self.scope,
            "source": self.source,
            "reason": self.reason,
            "released": self.released,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DerivedSummary:
    id: str
    conversation_id: str
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SummaryDecision:
    summary_id: str
    allowed: bool
    reason: str
    source: str

    def as_dict(self) -> dict:
        return {
            "summary_id": self.summary_id,
            "allowed": self.allowed,
            "reason": self.reason,
            "source": self.source,
        }


@dataclass(frozen=True)
class SensitivityReport:
    scope_query: str
    decisions: tuple[SensitivityDecision, ...]
    blocked_ids: tuple[str, ...]
    deleted_ids: tuple[str, ...]
    released_ids: tuple[str, ...]
    removed_from_board: tuple[str, ...]
    omitted_history: tuple[str, ...]
    summaries: tuple[SummaryDecision, ...]
    events: tuple[PreferenceEvent, ...]

    def allowed_ids(self) -> frozenset[str]:
        return frozenset(
            decision.message_id
            for decision in self.decisions
            if decision.allowed
        )

    def as_dict(self) -> dict:
        return {
            "decisions": [decision.as_dict() for decision in self.decisions],
            "blocked_ids": list(self.blocked_ids),
            "deleted_ids": list(self.deleted_ids),
            "released_ids": list(self.released_ids),
            "removed_from_board": list(self.removed_from_board),
            "omitted_history": list(self.omitted_history),
            "summaries": [summary.as_dict() for summary in self.summaries],
            "events": [event.as_dict() for event in self.events],
        }


class SensitivityPolicy:
    def __init__(self) -> None:
        self._preferences: dict[tuple[str, str, str], Preference] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"pref:{self._counter}"

    def _effective(self, conversation_id: str) -> tuple[Preference, ...]:
        return tuple(
            preference
            for (cid, _, _), preference in self._preferences.items()
            if cid == conversation_id
        )

    def _has_declared(self, conversation_id: str, scope: str, target: str) -> bool:
        preference = self._preferences.get((conversation_id, scope, target))
        return preference is not None and preference.source == SOURCE_DECLARED

    def declare(self, conversation_id: str, text: str, turn: int) -> tuple[PreferenceEvent, ...]:
        events: list[PreferenceEvent] = []
        normalized = normalize_text(text)
        for pattern, action in (
            *((pattern, ACTION_DELETE) for pattern in _DELETE_PATTERNS),
            *((pattern, ACTION_ALLOW) for pattern in _ALLOW_PATTERNS),
            *((pattern, ACTION_RESTRICT) for pattern in _RESTRICT_PATTERNS),
        ):
            match = re.search(pattern, normalized)
            if not match:
                continue
            classified = _classify(match.group("target"))
            if classified is None:
                continue
            scope, target = classified
            if action == ACTION_ALLOW:
                existing = self._preferences.get((conversation_id, scope, target))
                if existing is not None and existing.action == ACTION_DELETE:
                    events.append(
                        PreferenceEvent(
                            preference_id=existing.id,
                            operation="ignored",
                            action=existing.action,
                            scope=scope,
                            source=SOURCE_DECLARED,
                            turn=turn,
                        )
                    )
                    continue
            preference_id = self._next_id()
            existing = self._preferences.get((conversation_id, scope, target))
            if existing is not None:
                preference_id = existing.id
                operation = "updated"
            else:
                operation = "created"
            preference = Preference(
                id=preference_id,
                conversation_id=conversation_id,
                scope=scope,
                target=target,
                action=action,
                source=SOURCE_DECLARED,
                confidence=1.0,
                created_turn=existing.created_turn if existing else turn,
                updated_turn=turn,
            )
            self._preferences[(conversation_id, scope, target)] = preference
            events.append(
                PreferenceEvent(
                    preference_id=preference_id,
                    operation=operation,
                    action=action,
                    scope=scope,
                    source=SOURCE_DECLARED,
                    turn=turn,
                )
            )
        return tuple(events)

    def infer(
        self,
        conversation_id: str,
        text: str,
        turn: int,
        dominant_theme: str | None = None,
    ) -> tuple[PreferenceEvent, ...]:
        events: list[PreferenceEvent] = []
        normalized = normalize_text(text)
        for pattern in _INFERENCE_PATTERNS:
            match = re.search(pattern, normalized)
            if not match:
                continue
            raw_target = match.groupdict().get("target") or dominant_theme or ""
            classified = _classify(raw_target) if raw_target else None
            if classified is None and dominant_theme:
                classified = SCOPE_SUBJECT, dominant_theme
            if classified is None:
                continue
            scope, target = classified
            if self._has_declared(conversation_id, scope, target):
                events.append(
                    PreferenceEvent(
                        preference_id=self._preferences[
                            (conversation_id, scope, target)
                        ].id,
                        operation="ignored",
                        action=ACTION_RESTRICT,
                        scope=scope,
                        source=SOURCE_INFERENCE,
                        turn=turn,
                    )
                )
                continue
            existing = self._preferences.get((conversation_id, scope, target))
            preference_id = existing.id if existing else self._next_id()
            operation = "updated" if existing else "created"
            preference = Preference(
                id=preference_id,
                conversation_id=conversation_id,
                scope=scope,
                target=target,
                action=ACTION_RESTRICT,
                source=SOURCE_INFERENCE,
                confidence=0.7,
                created_turn=existing.created_turn if existing else turn,
                updated_turn=turn,
            )
            self._preferences[(conversation_id, scope, target)] = preference
            events.append(
                PreferenceEvent(
                    preference_id=preference_id,
                    operation=operation,
                    action=ACTION_RESTRICT,
                    scope=scope,
                    source=SOURCE_INFERENCE,
                    turn=turn,
                )
            )
        return tuple(events)

    def _matches(self, preference: Preference, message: Message) -> bool:
        if preference.scope == SCOPE_MESSAGE:
            return message.id == preference.target
        tokens = _message_tokens(message)
        if preference.scope == SCOPE_SUBJECT:
            lexicon = THEME_LEXICONS.get(preference.target)
            if lexicon is None:
                return bool(tokens & tokenize(preference.target))
            return bool(tokens & lexicon)
        target_tokens = tokenize(preference.target)
        return bool(target_tokens) and target_tokens <= tokens

    def _declares_for(self, preference: Preference, query: str) -> bool:
        normalized = normalize_text(query)
        for pattern in (*_DELETE_PATTERNS, *_RESTRICT_PATTERNS):
            match = re.search(pattern, normalized)
            if not match:
                continue
            classified = _classify(match.group("target"))
            if classified == (preference.scope, preference.target):
                return True
        return False

    def _release_matches(self, preference: Preference, query: str) -> bool:
        if preference.action == ACTION_DELETE:
            return False
        if self._declares_for(preference, query):
            return False
        normalized = normalize_text(query)
        if not RELEASE_MARKER.search(normalized):
            return False
        query_tokens = tokenize(query)
        if preference.scope == SCOPE_MESSAGE:
            return preference.target in re.findall(r"\bm\d+\b", normalized)
        if preference.scope == SCOPE_SUBJECT:
            lexicon = THEME_LEXICONS.get(preference.target, {preference.target})
            return bool(query_tokens & set(lexicon))
        target_tokens = tokenize(preference.target)
        return bool(target_tokens) and target_tokens <= query_tokens

    def evaluate(
        self,
        conversation_id: str,
        query: str,
        messages: Sequence[Message],
        turn: int,
    ) -> SensitivityReport:
        preferences = self._effective(conversation_id)
        decisions: list[SensitivityDecision] = []
        blocked: set[str] = set()
        deleted: set[str] = set()
        released: set[str] = set()

        for message in messages:
            matches = [
                preference for preference in preferences
                if self._matches(preference, message)
            ]
            if not matches:
                decisions.append(
                    SensitivityDecision(
                        message_id=message.id,
                        allowed=True,
                        action="nenhuma",
                        scope="",
                        source="",
                        reason="sem restrição",
                        released=False,
                        confidence=0.0,
                    )
                )
                continue
            deleting = [
                preference for preference in matches
                if preference.action == ACTION_DELETE
            ]
            if deleting:
                blocked.add(message.id)
                deleted.add(message.id)
                decisions.append(
                    SensitivityDecision(
                        message_id=message.id,
                        allowed=False,
                        action=ACTION_DELETE,
                        scope=deleting[0].scope,
                        source=deleting[0].source,
                        reason="exclusão solicitada",
                        released=False,
                        confidence=deleting[0].confidence,
                    )
                )
                continue
            restricting = [
                preference for preference in matches
                if preference.action == ACTION_RESTRICT
            ]
            if not restricting:
                decisions.append(
                    SensitivityDecision(
                        message_id=message.id,
                        allowed=True,
                        action=ACTION_ALLOW,
                        scope=matches[0].scope,
                        source=matches[0].source,
                        reason="permissão explícita",
                        released=False,
                        confidence=matches[0].confidence,
                    )
                )
                continue
            unreleased = [
                preference for preference in restricting
                if not self._release_matches(preference, query)
            ]
            if unreleased:
                blocked.add(message.id)
                chosen = sorted(
                    unreleased,
                    key=lambda preference: (
                        0 if preference.source == SOURCE_DECLARED else 1,
                        preference.scope,
                    ),
                )[0]
                reason = {
                    SCOPE_SUBJECT: "assunto restrito",
                    SCOPE_MESSAGE: "mensagem restrita",
                    SCOPE_DETAIL: "detalhe restrito",
                }[chosen.scope]
                decisions.append(
                    SensitivityDecision(
                        message_id=message.id,
                        allowed=False,
                        action=ACTION_RESTRICT,
                        scope=chosen.scope,
                        source=chosen.source,
                        reason=reason,
                        released=False,
                        confidence=chosen.confidence,
                    )
                )
            else:
                released.add(message.id)
                decisions.append(
                    SensitivityDecision(
                        message_id=message.id,
                        allowed=True,
                        action=ACTION_RESTRICT,
                        scope=restricting[0].scope,
                        source=restricting[0].source,
                        reason="liberado por pedido claro",
                        released=True,
                        confidence=restricting[0].confidence,
                    )
                )

        return SensitivityReport(
            scope_query=query,
            decisions=tuple(decisions),
            blocked_ids=tuple(sorted(blocked, key=lambda item: item)),
            deleted_ids=tuple(sorted(deleted, key=lambda item: item)),
            released_ids=tuple(sorted(released, key=lambda item: item)),
            removed_from_board=(),
            omitted_history=(),
            summaries=(),
            events=(),
        )

    def summary_is_blocked(
        self, conversation_id: str, summary: DerivedSummary, query: str
    ) -> tuple[bool, str]:
        preferences = self._effective(conversation_id)
        summary_tokens = tokenize(summary.text)
        for preference in preferences:
            if preference.action == ACTION_ALLOW:
                continue
            if preference.scope == SCOPE_MESSAGE:
                if preference.target in summary.source_ids:
                    if preference.action == ACTION_DELETE:
                        return True, "exclusão solicitada"
                    if not self._release_matches(preference, query):
                        return True, "mensagem restrita"
                    continue
            if preference.scope == SCOPE_SUBJECT:
                lexicon = THEME_LEXICONS.get(preference.target, {preference.target})
                matched = bool(summary_tokens & set(lexicon))
            else:
                target_tokens = tokenize(preference.target)
                matched = bool(target_tokens) and target_tokens <= summary_tokens
            if not matched:
                continue
            if preference.action == ACTION_DELETE:
                return True, "exclusão solicitada"
            if not self._release_matches(preference, query):
                return True, (
                    "assunto restrito" if preference.scope == SCOPE_SUBJECT
                    else "detalhe restrito"
                )
        return False, "sem restrição"

    def text_is_blocked(
        self, conversation_id: str, text: str, query: str
    ) -> tuple[bool, str]:
        tokens = tokenize(text)
        for preference in self._effective(conversation_id):
            if preference.action == ACTION_ALLOW:
                continue
            if preference.scope == SCOPE_SUBJECT:
                lexicon = THEME_LEXICONS.get(preference.target, {preference.target})
                matched = bool(tokens & set(lexicon))
            else:
                target_tokens = tokenize(preference.target)
                matched = bool(target_tokens) and target_tokens <= tokens
            if not matched:
                continue
            if preference.action == ACTION_DELETE:
                return True, "exclusão solicitada"
            if not self._release_matches(preference, query):
                return True, "conteúdo restrito"
        return False, "sem restrição"

    def with_context(
        self,
        report: SensitivityReport,
        *,
        removed_from_board: Iterable[str] = (),
        omitted_history: Iterable[str] = (),
        summaries: Sequence[SummaryDecision] = (),
        events: Sequence[PreferenceEvent] = (),
    ) -> SensitivityReport:
        return SensitivityReport(
            scope_query=report.scope_query,
            decisions=report.decisions,
            blocked_ids=report.blocked_ids,
            deleted_ids=report.deleted_ids,
            released_ids=report.released_ids,
            removed_from_board=tuple(removed_from_board),
            omitted_history=tuple(omitted_history),
            summaries=tuple(summaries),
            events=tuple(events),
        )
