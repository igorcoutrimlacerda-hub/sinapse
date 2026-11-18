"""Utilidade marginal de uma memória para o turno atual.

A estimativa combina três sinais determinísticos:

- evidência na dimensão pedida;
- associação com o acontecimento da semente;
- novidade em relação ao histórico recente e às evidências já escolhidas.

Afirmações repetidas têm ganho baixo. Números, datas, negação ou outro
acontecimento preservam a novidade, porque podem mudar a resposta. O ganho é
recalculado depois de cada inclusão.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .budget import document_signature, is_duplicate
from .memory import message_tokens
from .models import Message
from .text import normalize_text, tokenize


@dataclass(frozen=True)
class UtilityEstimate:
    gain: float
    evidence: float
    association: float
    novelty: float
    same_assertion: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "gain": self.gain,
            "evidence": self.evidence,
            "association": self.association,
            "novelty": self.novelty,
            "same_assertion": self.same_assertion,
            "reason": self.reason,
        }


def _signature_conflict(message: Message, other_text: str) -> bool:
    """Diferença observável que pode mudar a resposta (número/data/negação)."""
    signature = document_signature(message)
    other_normalized = normalize_text(other_text)
    other_message = Message(
        id="__history__",
        text=other_text,
        author="user",
        conversation_id=message.conversation_id,
        position=0,
    )
    other_signature = document_signature(other_message)
    if signature[0] != other_signature[0]:  # negação
        return True
    if signature[1] != other_signature[1]:  # números
        return True
    if signature[2] != other_signature[2]:  # datas
        return True
    return False


def estimate_utility(
    message: Message,
    evidence_focus: float | None,
    association: float,
    recent_turns: Sequence[str] = (),
    selected: Sequence[Message] = (),
    redundancy_weight: float = 0.6,
) -> UtilityEstimate:
    if evidence_focus is None:
        return UtilityEstimate(
            gain=0.0,
            evidence=0.0,
            association=float(association),
            novelty=0.0,
            same_assertion=False,
            reason="sem evidência no foco",
        )
    tokens = message_tokens(message)
    max_coverage = 0.0
    same_assertion = False
    signature_conflict = False
    if tokens:
        for text in recent_turns:
            other = tokenize(text)
            if not other:
                continue
            coverage = len(tokens & other) / len(tokens)
            max_coverage = max(max_coverage, coverage)
            if normalize_text(text) == normalize_text(message.text):
                same_assertion = True
            if coverage >= 0.5 and _signature_conflict(message, text):
                signature_conflict = True
        for other_message in selected:
            other_tokens = message_tokens(other_message)
            if other_tokens:
                coverage = len(tokens & other_tokens) / len(tokens)
                max_coverage = max(max_coverage, coverage)
                if coverage >= 0.5 and document_signature(message) != document_signature(
                    other_message
                ):
                    signature_conflict = True
            if is_duplicate(message, other_message):
                same_assertion = True

    novelty = 1.0 - min(1.0, max_coverage)
    if signature_conflict and not same_assertion:
        novelty = max(novelty, 0.65)
    base = max(0.0, float(evidence_focus)) * max(0.0, float(association))
    weight = min(1.0, max(0.0, float(redundancy_weight)))
    gain = base * ((1.0 - weight) + weight * novelty)
    if same_assertion:
        gain *= 0.1
        reason = "mesma afirmação já presente no contexto"
    elif signature_conflict:
        reason = "diferença preservada (negação, número, data ou acontecimento)"
    elif max_coverage >= 0.6:
        reason = "pouco acréscimo: conteúdo já coberto pelo contexto"
    else:
        reason = "novo detalhe para o pedido"
    return UtilityEstimate(
        gain=round(gain, 6),
        evidence=round(float(evidence_focus), 6),
        association=round(float(association), 6),
        novelty=round(novelty, 6),
        same_assertion=same_assertion,
        reason=reason,
    )
