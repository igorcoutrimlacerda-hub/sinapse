"""Fonte de candidatos para correções: busca revisões pela entidade e decisão.

A fonte só é acionada quando a pergunta pede a decisão vigente ou expõe
conflito entre versões (troca, mudança, correção, versão atual). Ela consulta
as mensagens de revisão que mencionam a entidade do pedido e filtra pela
decisão associada. A ordenação usa raridade (IDF) para preferir a revisão
específica em vez das revisões genéricas repetidas no corpus.

Restrições de sensibilidade desligam a fonte: nenhum candidato é injetado
quando a consulta declara uma preferência de restrição.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import Message
from .text import tokenize

# Verbos e expressões que marcam uma revisão dentro de uma mensagem.
REVISION_MARKERS: frozenset[str] = frozenset(
    {
        "troquei",
        "troque",
        "trocar",
        "trocado",
        "mudei",
        "mude",
        "mudar",
        "mudado",
        "substitui",
        "substituir",
        "corrigi",
        "corrigir",
        "correcao",
        "atualizei",
        "atualizar",
        "verdade",
        "vigente",
        "versao",
    }
)

# Sinais na pergunta de que o pedido é pela decisão vigente/conflito.
ACTIVATION_MARKERS: frozenset[str] = frozenset(
    {
        "troquei",
        "troque",
        "trocar",
        "trocado",
        "mudei",
        "mude",
        "mudar",
        "mudado",
        "substitui",
        "substituir",
        "corrigi",
        "corrigir",
        "correcao",
        "atual",
        "atualizei",
        "atualizar",
        "vigente",
        "versao",
        "verdade",
    }
)

# Palavras de pergunta que não identificam a decisão.
GENERIC_TERMS: frozenset[str] = frozenset(
    {
        "quanto",
        "quantos",
        "quantas",
        "onde",
        "porque",
        "depois",
        "antes",
        "agora",
        "entao",
        "afinal",
        "mesmo",
        "ainda",
    }
)

_NAME = re.compile(r"\b[A-ZÁÉÍÓÚÂÊÎÔÃÕÇ][a-záéíóúâêîôãõç]{2,}\b")


@dataclass(frozen=True)
class CorrectionSelection:
    activated: bool
    reason: str
    indices: tuple[int, ...] = ()
    entity_terms: tuple[str, ...] = ()
    decision_terms: tuple[str, ...] = ()
    matched: int = 0
    elapsed_ms: float = 0.0


def _idf(postings: Mapping[str, Sequence[int]], token: str, total: int) -> float:
    frequency = len(postings.get(token, ()))
    return math.log((total + 1) / (frequency + 1)) + 1.0


def _entity_terms(question: str, history: Sequence[str]) -> tuple[str, ...]:
    terms: set[str] = set()
    for text in (question, *history):
        for match in _NAME.finditer(text):
            if match.start() == 0:
                # Primeira palavra da frase costuma ser verbo/saudação.
                continue
            terms.update(tokenize(match.group(0)))
    return tuple(sorted(terms))


def _decision_terms(
    question: str, entity_terms: Sequence[str]
) -> tuple[str, ...]:
    tokens = tokenize(question)
    excluded = set(REVISION_MARKERS) | set(ACTIVATION_MARKERS) | set(GENERIC_TERMS)
    excluded.update(entity_terms)
    return tuple(sorted(tokens - excluded))


def find_corrections(
    messages: Sequence[Message],
    message_tokens: Mapping[int, frozenset[str]],
    postings: Mapping[str, Sequence[int]],
    question: str,
    history: Sequence[str] = (),
    preferences: Sequence[str] = (),
    limit: int = 3,
) -> CorrectionSelection:
    """Retorna os índices das revisões mais específicas para o pedido."""

    started = time.perf_counter()
    if preferences:
        return CorrectionSelection(
            False,
            "sensibilidade",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    question_tokens = tokenize(question)
    if not question_tokens & ACTIVATION_MARKERS:
        return CorrectionSelection(
            False,
            "sem_sinal_de_correcao",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    entity_terms = _entity_terms(question, history)
    decision_terms = _decision_terms(question, entity_terms)
    if not entity_terms and not decision_terms:
        return CorrectionSelection(
            False,
            "sem_entidade_ou_decisao",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    marker_indices: set[int] = set()
    for marker in REVISION_MARKERS:
        marker_indices.update(postings.get(marker, ()))
    if entity_terms:
        entity_indices: set[int] = set()
        for term in entity_terms:
            entity_indices.update(postings.get(term, ()))
        pool = marker_indices & entity_indices
    else:
        pool = marker_indices
    if decision_terms:
        pool = {
            index
            for index in pool
            if message_tokens[index] & set(decision_terms)
        }

    total = len(messages)
    ranked: list[tuple[float, int]] = []
    for index in pool:
        tokens = message_tokens[index]
        entity_hits = sum(
            _idf(postings, term, total)
            for term in entity_terms
            if term in tokens
        )
        decision_hits = sum(
            _idf(postings, term, total)
            for term in decision_terms
            if term in tokens
        )
        specificity = max(
            (_idf(postings, term, total) for term in tokens), default=0.0
        )
        score = 2.0 * entity_hits + 2.0 * decision_hits + 0.5 * specificity
        ranked.append((score, index))
    ranked.sort(
        key=lambda item: (item[0], messages[item[1]].position), reverse=True
    )
    indices = tuple(index for _, index in ranked[:limit])
    return CorrectionSelection(
        True,
        "ativo" if indices else "sem_candidatos",
        indices=indices,
        entity_terms=entity_terms,
        decision_terms=decision_terms,
        matched=len(pool),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
