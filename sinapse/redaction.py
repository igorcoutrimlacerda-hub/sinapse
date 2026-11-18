"""Redação de credenciais antes de qualquer envio ou log."""

from __future__ import annotations

import re

PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password|senha)"
        r"\s*[:=]\s*\S{6,}"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
PLACEHOLDER = "[credencial removida]"


def contains_credential(text: str) -> bool:
    return any(pattern.search(str(text)) for pattern in PATTERNS)


def redact_text(text: str) -> str:
    value = str(text)
    for pattern in PATTERNS:
        value = pattern.sub(PLACEHOLDER, value)
    return value


def redact_any(value):
    """Redige strings dentro de listas/dicionários de payload."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_any(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_any(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_any(item) for key, item in value.items()}
    return value
