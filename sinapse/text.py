"""Normalização de texto, tokens, inferência de dimensão e referências."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

DIMENSIONS: tuple[str, ...] = (
    "o_que",
    "quem",
    "onde",
    "quando",
    "como",
    "por_que",
    "para_que",
)

# Palavras funcionais. "nao" fica de fora de propósito: negação é conteúdo.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "o", "as", "os", "um", "uma", "uns", "umas",
        "de", "da", "do", "das", "dos", "para", "pra", "por", "que",
        "e", "com", "em", "no", "na", "nos", "nas", "ao", "aos",
        "eu", "ele", "ela", "eles", "elas", "nos", "vos",
        "isso", "isto", "esse", "essa", "este", "esta", "aquilo", "aquele", "aquela",
        "como", "quem", "onde", "quando", "qual", "quais", "porque", "porquê",
        "ja", "la", "se", "me", "te", "the", "of", "to", "and", "is", "it",
        "this", "that", "an", "in", "on", "at", "was", "were",
    }
)

PRONOUNS: frozenset[str] = frozenset(
    {"ele", "ela", "eles", "elas", "isso", "isto", "esse", "essa", "este", "esta", "aquilo", "aquele", "aquela"}
)

NEGATIONS: frozenset[str] = frozenset({"nao", "nunca", "jamais", "nem"})

_MONTHS = (
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, pontuação colapsada em espaço."""
    plain = strip_accents(text.lower())
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def tokenize(text: str) -> frozenset[str]:
    """Tokens de conteúdo determinísticos (palavras/números sem stopwords)."""
    plain = normalize_text(text)
    return frozenset(t for t in plain.split() if t and t not in STOPWORDS)


def tokenize_values(values: Iterable[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(tokenize(value))
    return frozenset(tokens)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / ((len(left) * len(right)) ** 0.5)


# Ordem importa: "para que" antes de "por que"/"o que", "por que" antes de "o que".
_DIMENSION_RULES: tuple[tuple[str, str], ...] = (
    ("para_que", r"\bpara\s+que\b|\bpra\s+que\b|\bobjetivo\b|\bfinalidade\b"),
    ("por_que", r"\bpor\s+que\b|\bporque\b|\bporqu[eê]\b|\bmotivo\b|\braz[aã]o\b|\bpor\s+qual\s+razao\b"),
    ("o_que", r"\bo\s+que\b|\bque\s+foi\b"),
    ("como", r"\bcomo\b|\bprocedimento\b|\bpasso\s+a\s+passo\b"),
    ("quem", r"\bquem\b"),
    ("onde", r"\bonde\b|\blugar\b"),
    ("quando", r"\bquando\b|\bque\s+dia\b|\bque\s+horas\b"),
)


def infer_dimension(text: str) -> tuple[str, str]:
    """Retorna (dimensão, regra textual acionada). Fallback: o_que."""
    plain = strip_accents(text.lower())
    for dimension, pattern in _DIMENSION_RULES:
        match = re.search(pattern, plain)
        if match:
            return dimension, match.group(0)
    return "o_que", "fallback (sem palavra interrogativa conhecida)"


@dataclass(frozen=True)
class ReferenceResolution:
    pronouns: tuple[str, ...] = ()
    source_turn_index: int | None = None
    referent_tokens: tuple[str, ...] = ()
    note: str = "sem pronome anafórico"

    def as_dict(self) -> dict:
        return {
            "pronouns": list(self.pronouns),
            "source_turn_index": self.source_turn_index,
            "referent_tokens": list(self.referent_tokens),
            "note": self.note,
        }


def find_pronouns(text: str) -> tuple[str, ...]:
    words = re.findall(r"[a-z]+", strip_accents(text.lower()))
    return tuple(dict.fromkeys(w for w in words if w in PRONOUNS))


def resolve_references(current_text: str, recent_turns: Sequence[str]) -> ReferenceResolution:
    """Resolução rasa: pronomes apontam para o turno anterior mais próximo.

    O âncora da consulta sempre inclui os turnos recentes; esta resolução
    registra o antecedente e seus termos para inspeção e para o diagnóstico.
    """
    pronouns = find_pronouns(current_text)
    if not pronouns:
        return ReferenceResolution()
    for index in range(len(recent_turns) - 1, -1, -1):
        referents = tokenize(recent_turns[index])
        if referents:
            return ReferenceResolution(
                pronouns=pronouns,
                source_turn_index=index,
                referent_tokens=tuple(sorted(referents)),
                note="pronome resolvido pelo turno anterior mais próximo com conteúdo",
            )
    return ReferenceResolution(
        pronouns=pronouns,
        note="pronome encontrado, mas não havia turno anterior com conteúdo",
    )


def query_anchor(query_text: str, recent_turns: Sequence[str]) -> frozenset[str]:
    anchor: set[str] = set(tokenize(query_text))
    for turn in recent_turns:
        anchor.update(tokenize(turn))
    return frozenset(anchor)


def negation_present(text: str) -> bool:
    return bool(tokenize(text) & NEGATIONS)


def number_signature(text: str) -> tuple[str, ...]:
    plain = normalize_text(text)
    return tuple(sorted(re.findall(r"\b\d+\b", plain)))


def date_signature(text: str) -> tuple[str, ...]:
    plain = normalize_text(text)
    numeric = re.findall(r"\b\d{1,4}\s*[/.-]\s*\d{1,2}(?:\s*[/.-]\s*\d{1,4})?\b", plain)
    months = re.findall(r"\b(?:" + "|".join(_MONTHS) + r")\b", plain)
    return tuple(sorted(set(numeric + months)))
