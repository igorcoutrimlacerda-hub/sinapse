"""Atributos pedidos pelo turno e evidência específica.

A dimensão ampla (`o_que`) não basta: uma pergunta de preço exige que alguma
mensagem registre preço. Tema, distância e associação continuam valendo, mas
são condições adicionais, não substitutas do atributo.

Nesta versão o atributo verificável é `preco`. Ausência de preço em todas as
mensagens faz o contexto adicional ficar vazio; proximidade temática não é
tratada como evidência de preço.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from .models import Message
from .text import normalize_text, tokenize, tokenize_values

ATTRIBUTE_PRICE = "preco"

_QUERY_PRICE = re.compile(
    r"\b(?:quanto\s+custou|quanto\s+custa|quanto\s+foi|qual\s+foi\s+o\s+preco|"
    r"qual\s+o\s+preco|preco|valor|paguei|pagou|custou|custa)\b"
)

_ATTRIBUTE_QUERY_TERMS = frozenset(
    {
        "quanto",
        "custou",
        "custa",
        "preco",
        "valor",
        "paguei",
        "pagou",
        "qual",
        "foi",
        "me",
        "diz",
        "diga",
        "informa",
        "informe",
    }
)

# Preço precisa de moeda/valor ou de um termo de preço junto de um número.
_PRICE_WITH_CURRENCY = re.compile(
    r"(?:\br\s+\d+(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?\s*(?:reais|real|brl)\b)"
)
_PRICE_WITH_TERM = re.compile(
    r"\b(?:preco|valor|custou|custa|paguei|pagou)\b[^.]{0,30}\b\d+(?:[.,]\d+)?\b"
)


def requested_attribute(query: str) -> str | None:
    normalized = normalize_text(query)
    if _QUERY_PRICE.search(normalized):
        return ATTRIBUTE_PRICE
    return None


def attribute_targets(query: str) -> frozenset[str]:
    """Termos do item pedido, sem as palavras do atributo."""
    return frozenset(tokenize(query) - _ATTRIBUTE_QUERY_TERMS)


def _message_tokens(message: Message) -> frozenset[str]:
    tokens: set[str] = set(tokenize(message.text))
    for dimension in message.coordinates:
        tokens.update(tokenize_values(message.values(dimension)))
    return frozenset(tokens)


def is_price_evidence(message: Message) -> bool:
    parts = [message.text]
    for dimension in message.coordinates:
        parts.extend(message.values(dimension))
    text = normalize_text(" ".join(parts))
    return bool(_PRICE_WITH_CURRENCY.search(text) or _PRICE_WITH_TERM.search(text))


def message_supports_attribute(
    message: Message,
    attribute: str,
    targets: frozenset[str] | None = None,
    by_id: Mapping[str, Message] | None = None,
) -> bool:
    if attribute != ATTRIBUTE_PRICE:
        return True
    if not is_price_evidence(message):
        return False
    if not targets:
        return True
    if _message_tokens(message) & targets:
        return True
    if by_id and message.reply_to:
        parent = by_id.get(message.reply_to)
        if parent is not None and _message_tokens(parent) & targets:
            return True
    return False


def supporting_ids(
    messages: Sequence[Message],
    attribute: str,
    conversation_id: str | None = None,
    targets: frozenset[str] | None = None,
    by_id: Mapping[str, Message] | None = None,
) -> frozenset[str]:
    return frozenset(
        message.id
        for message in messages
        if (conversation_id is None or message.conversation_id == conversation_id)
        and message_supports_attribute(message, attribute, targets, by_id)
    )
