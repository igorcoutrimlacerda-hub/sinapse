"""Atenção temática determinística por turno.

O relatório temático é produzido antes da evocação e orienta três coisas:

- a busca de semente (viés por tema no núcleo de evocação);
- a ordem e o corte das evidências no quadro branco;
- o diagnóstico local (pesos, sinais e ambiguidade).

Regras: menção explícita no turno atual vale 1.0 e reativa o tema; temas não
mencionados decaem por um fator fixo; temas com apoio no histórico/quadro
mantêm um piso; dois temas próximos sem menção explícita tornam o turno
ambíguo, e o quadro reduz o número de evidências.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import Message
from .text import infer_dimension, tokenize, tokenize_values

DECAY = 0.55
MIN_WEIGHT = 0.20
HISTORY_SUPPORT = 0.40
EXPLICIT_WEIGHT = 1.0
AMBIGUITY_RATIO = 0.85

THEME_LEXICONS: Mapping[str, frozenset[str]] = {
    "serum": frozenset({"serum", "preset", "presets", "sintetizador", "synth", "plugin"}),
    "bass": frozenset({"bass", "baixo", "grave", "graves", "envelope", "envelopes", "sub"}),
    "xadrez": frozenset({"xadrez", "jogando", "jogar", "partida", "tabuleiro"}),
    "piano": frozenset({"piano", "jazz"}),
    "pack": frozenset(
        {
            "pack",
            "pacote",
            "presets",
            "divulgacao",
            "divulgar",
            "divulguei",
            "divulgou",
            "forum",
            "lancamento",
        }
    ),
}


@dataclass(frozen=True)
class ThemeSignal:
    theme: str
    weight: float
    previous_weight: float
    explicit: bool
    support: str
    matched_terms: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "theme": self.theme,
            "weight": self.weight,
            "previous_weight": self.previous_weight,
            "explicit": self.explicit,
            "support": self.support,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class ThemeReport:
    themes: tuple[ThemeSignal, ...]
    dominant: str | None
    requested_dimension: str
    dimension_rule: str
    ambiguous: bool
    ambiguity_reason: str | None
    signals: tuple[str, ...]
    decay: float

    def weights(self) -> dict[str, float]:
        return {signal.theme: signal.weight for signal in self.themes}

    @property
    def explicit_themes(self) -> tuple[str, ...]:
        return tuple(signal.theme for signal in self.themes if signal.explicit)

    @property
    def dimension_explicit(self) -> bool:
        return not self.dimension_rule.startswith("fallback")

    def as_dict(self) -> dict:
        return {
            "themes": [signal.as_dict() for signal in self.themes],
            "dominant": self.dominant,
            "requested_dimension": self.requested_dimension,
            "dimension_rule": self.dimension_rule,
            "ambiguous": self.ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "signals": list(self.signals),
            "decay": self.decay,
        }


def _tokens_for_message(message: Message) -> frozenset[str]:
    tokens: set[str] = set(tokenize(message.text))
    for dimension in message.coordinates:
        tokens.update(tokenize_values(message.values(dimension)))
    return frozenset(tokens)


def _effective_tokens(
    message: Message,
    by_id: Mapping[str, Message] | None = None,
) -> frozenset[str]:
    """Tokens do ponto; uma resposta curta herda o assunto do pai (1 salto).

    É o mesmo princípio do vínculo de resposta usado na distância: a licença
    (m2) não repete "Serum"/"bass", mas pertence ao assunto de m1. A herança só
    ocorre quando o texto próprio não traz nenhum termo temático.
    """
    tokens = _tokens_for_message(message)
    if not by_id or not message.reply_to:
        return tokens
    if any(tokens & lexicon for lexicon in THEME_LEXICONS.values()):
        return tokens
    parent = by_id.get(message.reply_to)
    if parent is None:
        return tokens
    return _tokens_for_message(parent)


def theme_affinity(
    message: Message,
    weights: Mapping[str, float],
    by_id: Mapping[str, Message] | None = None,
) -> float:
    """Fração do peso temático ativo coberta pela mensagem (0..1)."""
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    tokens = _effective_tokens(message, by_id)
    matched = sum(
        weight for theme, weight in weights.items() if tokens & THEME_LEXICONS[theme]
    )
    return max(0.0, min(1.0, matched / total))


def themes_for_message(
    message: Message,
    weights: Mapping[str, float],
    by_id: Mapping[str, Message] | None = None,
) -> tuple[str, ...]:
    tokens = _effective_tokens(message, by_id)
    matched = [
        theme for theme in weights if tokens & THEME_LEXICONS[theme]
    ]
    return tuple(sorted(matched, key=lambda theme: (-weights[theme], theme)))


def analyze_turn(
    query: str,
    recent_turns: Sequence[str] = (),
    previous_weights: Mapping[str, float] | None = None,
    board_texts: Sequence[str] = (),
    decay: float = DECAY,
) -> ThemeReport:
    previous = dict(previous_weights or {})
    query_tokens = tokenize(query)
    history_tokens = tokenize(" ".join(recent_turns))
    board_tokens = tokenize(" ".join(board_texts))
    dimension, dimension_rule = infer_dimension(query)

    explicit: dict[str, tuple[str, ...]] = {}
    for theme, lexicon in THEME_LEXICONS.items():
        hits = tuple(sorted(query_tokens & lexicon))
        if hits:
            explicit[theme] = hits

    raw: dict[str, float] = {}
    support: dict[str, str] = {}
    matched_terms: dict[str, tuple[str, ...]] = {}
    for theme, lexicon in THEME_LEXICONS.items():
        previous_weight = float(previous.get(theme, 0.0))
        decayed = previous_weight * decay
        if theme in explicit:
            raw[theme] = EXPLICIT_WEIGHT
            support[theme] = "turno_atual"
            matched_terms[theme] = explicit[theme]
            continue
        if decayed > 0:
            has_history = bool((history_tokens | board_tokens) & lexicon)
            raw[theme] = max(decayed, HISTORY_SUPPORT) if has_history else decayed
            support[theme] = "historico_recente" if has_history else "decaimento"
            continue
        if (history_tokens | board_tokens) & lexicon:
            raw[theme] = HISTORY_SUPPORT
            support[theme] = "historico_recente"

    raw = {theme: value for theme, value in raw.items() if value >= MIN_WEIGHT}
    peak = max(raw.values()) if raw else 0.0
    weights = {
        theme: round(value / peak, 4) if peak > 0 else 0.0
        for theme, value in raw.items()
    }
    ordered = tuple(
        ThemeSignal(
            theme=theme,
            weight=weights[theme],
            previous_weight=round(float(previous.get(theme, 0.0)), 4),
            explicit=theme in explicit,
            support=support[theme],
            matched_terms=matched_terms.get(theme, ()),
        )
        for theme in sorted(weights, key=lambda item: (-weights[item], item))
    )

    signals: list[str] = []
    for signal in ordered:
        if signal.explicit:
            signals.append(
                f"menção explícita: {signal.theme} ({', '.join(signal.matched_terms)})"
            )
        elif signal.support == "historico_recente":
            signals.append(
                f"apoio no histórico/quadro: {signal.theme} "
                f"({signal.previous_weight:.3f} -> {signal.weight:.3f})"
            )
        else:
            signals.append(
                f"decaimento: {signal.theme} "
                f"({signal.previous_weight:.3f} -> {signal.weight:.3f})"
            )
    if not explicit:
        signals.append("sem menção explícita no turno; interpretado pelo histórico/quadro")
    signals.append(f"dimensão pedida: {dimension} via {dimension_rule!r}")

    ambiguous = False
    ambiguity_reason: str | None = None
    if not explicit and len(ordered) >= 2:
        top, second = ordered[0], ordered[1]
        if top.weight > 0 and second.weight >= top.weight * AMBIGUITY_RATIO:
            ambiguous = True
            ambiguity_reason = (
                f"temas {top.theme} e {second.theme} com pesos próximos "
                f"({top.weight:.3f} e {second.weight:.3f}) e sem menção explícita"
            )
            signals.append(f"ambiguidade: {ambiguity_reason}")

    return ThemeReport(
        themes=ordered,
        dominant=ordered[0].theme if ordered else None,
        requested_dimension=dimension,
        dimension_rule=dimension_rule,
        ambiguous=ambiguous,
        ambiguity_reason=ambiguity_reason,
        signals=tuple(signals),
        decay=decay,
    )
