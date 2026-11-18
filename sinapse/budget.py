"""Orçamento de contexto: contagem, seleção e diagnóstico.

A contagem é sempre feita sobre o texto final renderizado (entrada-base +
separador + bloco de memória), incluindo rótulos e metadados. Distância,
vetor, atenção e motivos nunca entram no contexto entregue ao chatbot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .memory import requires_seed
from .models import Candidate, Message
from .text import (
    date_signature,
    normalize_text,
    negation_present,
    number_signature,
    tokenize_values,
)


class TokenCounter(Protocol):
    name: str
    simulated: bool

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class SimulatedCounter:
    """Contador determinístico de demonstração: palavras + pontuação."""

    name: str = "simulated-words-punct-v1"
    simulated: bool = True

    def count(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@dataclass(frozen=True)
class CharCounter:
    """Contador de caracteres para testes de encaixe exato."""

    name: str = "char-v1"
    simulated: bool = True

    def count(self, text: str) -> int:
        return len(text)


@dataclass(frozen=True)
class RecentTurn:
    text: str
    id: str | None = None
    author: str = "user"
    position: int | None = None


@dataclass(frozen=True)
class BaseInput:
    instructions: str = ""
    recent_turns: tuple[RecentTurn, ...] = ()
    current_turn: str = ""
    explicit_message_ids: tuple[str, ...] = ()

    def message_ids(self) -> set[str]:
        ids = {t.id for t in self.recent_turns if t.id}
        ids.update(self.explicit_message_ids)
        return ids

    def render(self) -> str:
        lines = ["[INSTRUÇÕES]", self.instructions or "(vazio)", "[HISTÓRICO RECENTE]"]
        if self.recent_turns:
            for turn in self.recent_turns:
                label = f"[{turn.author}"
                if turn.id:
                    label += f"|{turn.id}"
                if turn.position is not None:
                    label += f"|pos={turn.position}"
                label += "]"
                lines.append(f"{label} {turn.text}")
        else:
            lines.append("(vazio)")
        lines.extend(["[TURNO ATUAL]", self.current_turn])
        return "\n".join(lines)


def render_seed_block(seed: Message) -> str:
    reply = seed.reply_to or "-"
    return (
        f"[semente] id={seed.id} | autor={seed.author} | conversa={seed.conversation_id} "
        f"| pos={seed.position} | resposta_a={reply}\n{seed.text}"
    )


def render_evidence_block(message: Message, rank: int) -> str:
    reply = message.reply_to or "-"
    return (
        f"[evidência {rank}] id={message.id} | autor={message.author} "
        f"| conversa={message.conversation_id} | pos={message.position} | resposta_a={reply}\n"
        f"{message.text}"
    )


def wrap_memory(blocks: Sequence[str]) -> str:
    if not blocks:
        return ""
    return "=== MEMÓRIA (Sinapse) ===\n" + "\n\n".join(blocks) + "\n=== FIM DA MEMÓRIA ==="


def event_signature(message: Message) -> tuple[str, ...]:
    tokens: set[str] = set()
    for dimension in ("o_que", "onde", "quando", "para_que"):
        tokens.update(tokenize_values(message.values(dimension)))
    return tuple(sorted(tokens))


def document_signature(message: Message) -> tuple[bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        negation_present(message.text),
        number_signature(message.text),
        date_signature(message.text),
        event_signature(message),
    )


def is_duplicate(left: Message, right: Message) -> bool:
    """Deduplicação conservadora.

    Só declara equivalência quando o texto normalizado é idêntico E a
    assinatura (negação, números, datas e acontecimento) coincide. Similaridade
    alta isolada nunca basta; paráfrases e negações permanecem disponíveis.
    """
    if normalize_text(left.text) != normalize_text(right.text):
        return False
    return document_signature(left) == document_signature(right)


@dataclass(frozen=True)
class CandidateDecision:
    id: str
    text: str
    distance: float
    association: float
    association_content: float
    association_link: float
    association_source: str
    evidence: Mapping[str, float | None]
    attention: float
    within_radius: bool
    reason: str
    detail: str | None = None
    rank: int | None = None
    tokens: int | None = None
    sent_at: str | None = None
    event_time: tuple[str, ...] = ()
    reply_to: str | None = None
    marginal_gain: float | None = None
    selection_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "distance": self.distance,
            "association": self.association,
            "association_content": self.association_content,
            "association_link": self.association_link,
            "association_source": self.association_source,
            "evidence": dict(self.evidence),
            "attention": self.attention,
            "within_radius": self.within_radius,
            "reason": self.reason,
            "detail": self.detail,
            "rank": self.rank,
            "tokens": self.tokens,
            "sent_at": self.sent_at,
            "event_time": list(self.event_time),
            "reply_to": self.reply_to,
            "marginal_gain": self.marginal_gain,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class BudgetDiagnostics:
    counter_name: str
    counter_simulated: bool
    window: int
    output_reserve: int
    base_tokens: int
    capacity: int
    memory_budget: int
    memory_tokens: int
    used_tokens: int
    remaining_tokens: int
    total_with_reserve: int
    exceeds_capacity: bool
    status: str
    base_overflow_tokens: int
    seed_included: bool
    seed_in_base: bool
    seed_source: str
    seed_tokens: int
    selected_ids: tuple[str, ...]
    decisions: tuple[CandidateDecision, ...]
    context_format: str = "text"
    min_marginal_gain: float = 0.0
    stopped_early: bool = False
    stop_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "context_format": self.context_format,
            "counter": {"name": self.counter_name, "simulated": self.counter_simulated},
            "window": self.window,
            "output_reserve": self.output_reserve,
            "base_tokens": self.base_tokens,
            "capacity": self.capacity,
            "memory_budget": self.memory_budget,
            "memory_tokens": self.memory_tokens,
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "total_with_reserve": self.total_with_reserve,
            "exceeds_capacity": self.exceeds_capacity,
            "status": self.status,
            "base_overflow_tokens": self.base_overflow_tokens,
            "seed_included": self.seed_included,
            "seed_in_base": self.seed_in_base,
            "seed_source": self.seed_source,
            "seed_tokens": self.seed_tokens,
            "selected_ids": list(self.selected_ids),
            "decisions": [d.as_dict() for d in self.decisions],
            "min_marginal_gain": self.min_marginal_gain,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class BudgetResult:
    context_text: str
    memory_text: str
    diagnostics: BudgetDiagnostics

    def as_dict(self) -> dict:
        return {
            "context_text": self.context_text,
            "memory_text": self.memory_text,
            "diagnostics": self.diagnostics.as_dict(),
        }


def _decision_from(candidate: Candidate, reason: str, detail: str | None = None,
                   rank: int | None = None, tokens: int | None = None,
                   marginal_gain: float | None = None,
                   selection_reason: str | None = None) -> CandidateDecision:
    return CandidateDecision(
        id=candidate.message.id,
        text=candidate.message.text,
        distance=candidate.distance,
        association=candidate.association,
        association_content=candidate.association_content,
        association_link=candidate.association_link,
        association_source=candidate.association_source,
        evidence={d: candidate.evidence.get(d) for d in candidate.evidence},
        attention=candidate.attention,
        within_radius=candidate.within_radius,
        reason=reason,
        detail=detail,
        rank=rank,
        tokens=tokens,
        sent_at=candidate.message.sent_at,
        event_time=candidate.message.values("quando"),
        reply_to=candidate.message.reply_to,
        marginal_gain=marginal_gain,
        selection_reason=selection_reason,
    )


def _join(base_text: str, memory_text: str) -> str:
    return base_text + ("\n\n" + memory_text if memory_text else "")


def _fit(
    base_text: str,
    blocks: Sequence[str],
    counter: TokenCounter,
    window: int,
    output_reserve: int,
) -> tuple[int, int]:
    text = _join(base_text, wrap_memory(blocks))
    used = counter.count(text)
    return used, used + output_reserve - window


def select_context(
    base_input: BaseInput,
    seed: Message,
    focus: str,
    candidates: Sequence[Candidate],
    window: int,
    output_reserve: int,
    counter: TokenCounter,
    top_k: int = 3,
    threshold: float = 0.05,
    include_seed: bool = True,
) -> BudgetResult:
    base_text = base_input.render()
    base_tokens = int(counter.count(base_text))
    capacity = int(window) - int(output_reserve)
    memory_budget = max(0, capacity - base_tokens)
    base_ids = base_input.message_ids()
    seed_in_base = seed.id in base_ids

    if base_tokens > capacity:
        excess = base_tokens - capacity
        decisions = tuple(
            _decision_from(
                candidate,
                "base_overflow",
                f"entrada-base excede a capacidade em {excess} tokens; memória não avaliada",
            )
            for candidate in candidates
        )
        diagnostics = BudgetDiagnostics(
            counter_name=counter.name,
            counter_simulated=bool(counter.simulated),
            window=int(window),
            output_reserve=int(output_reserve),
            base_tokens=base_tokens,
            capacity=capacity,
            memory_budget=0,
            memory_tokens=0,
            used_tokens=base_tokens,
            remaining_tokens=capacity - base_tokens,
            total_with_reserve=base_tokens + int(output_reserve),
            exceeds_capacity=True,
            status="base_overflow",
            base_overflow_tokens=excess,
            seed_included=False,
            seed_in_base=seed_in_base,
            seed_source="not_evaluated",
            seed_tokens=0,
            selected_ids=(),
            decisions=decisions,
        )
        return BudgetResult(context_text=base_text, memory_text="", diagnostics=diagnostics)

    if memory_budget == 0:
        decisions = tuple(
            _decision_from(
                candidate,
                "outside_radius" if not candidate.within_radius else "no_budget",
                "orçamento de memória igual a zero" if candidate.within_radius else None,
            )
            for candidate in candidates
        )
        diagnostics = BudgetDiagnostics(
            counter_name=counter.name,
            counter_simulated=bool(counter.simulated),
            window=int(window),
            output_reserve=int(output_reserve),
            base_tokens=base_tokens,
            capacity=capacity,
            memory_budget=0,
            memory_tokens=0,
            used_tokens=base_tokens,
            remaining_tokens=capacity - base_tokens,
            total_with_reserve=base_tokens + int(output_reserve),
            exceeds_capacity=False,
            status="no_budget",
            base_overflow_tokens=0,
            seed_included=False,
            seed_in_base=seed_in_base,
            seed_source="base" if seed_in_base else "omitted",
            seed_tokens=0,
            selected_ids=(),
            decisions=decisions,
        )
        return BudgetResult(context_text=base_text, memory_text="", diagnostics=diagnostics)

    ordered = sorted(
        candidates,
        key=lambda c: (-c.attention, c.distance, c.message.position, c.message.id),
    )
    reasons: dict[str, tuple[str, str | None]] = {}
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    seed_required = False

    for candidate in ordered:
        message = candidate.message
        if not candidate.within_radius:
            reasons[message.id] = ("outside_radius", None)
            continue
        if candidate.evidence.get(focus) is None:
            reasons[message.id] = (
                "no_focus_evidence",
                f"sem informação própria em {focus}; atenção tratada como 0",
            )
            continue
        if candidate.attention <= threshold:
            reasons[message.id] = (
                "below_threshold",
                f"atenção {candidate.attention:.4f} <= limiar {threshold:.4f}",
            )
            continue
        if message.id in base_ids:
            reasons[message.id] = ("already_in_base", "ID já presente na entrada-base")
            continue
        if len(selected) >= top_k:
            reasons[message.id] = ("top_k_full", f"K={top_k} evidências já selecionadas")
            continue
        if any(is_duplicate(message, chosen.message) for chosen in selected):
            reasons[message.id] = ("duplicate", "mesma afirmação e mesmo acontecimento")
            continue

        needs_seed = requires_seed(message, seed) and not seed_in_base
        if needs_seed and not include_seed:
            reasons[message.id] = (
                "requires_seed",
                "resposta depende da semente e include_seed=False",
            )
            continue
        include_seed_now = seed_required or needs_seed
        blocks: list[str] = []
        if include_seed_now:
            blocks.append(render_seed_block(seed))
        for rank, chosen in enumerate(selected, start=1):
            blocks.append(render_evidence_block(chosen.message, rank))
        blocks.append(render_evidence_block(message, len(selected) + 1))

        used, deficit = _fit(base_text, blocks, counter, window, output_reserve)
        candidate_tokens = int(counter.count(render_evidence_block(message, 1)))
        if deficit <= 0:
            selected.append(candidate)
            selected_ids.add(message.id)
            if needs_seed:
                seed_required = True
            reasons[message.id] = (
                "selected",
                f"selecionada na posição {len(selected)}; conjunto com {used} tokens",
            )
        else:
            reasons[message.id] = (
                "does_not_fit",
                f"faltam {deficit} tokens para o conjunto inteiro",
            )

    seed_included = seed_required
    seed_source = "memory_required" if seed_required else ("base" if seed_in_base else "omitted")
    seed_tokens = 0
    if seed_included:
        seed_tokens = (
            int(counter.count(_join(base_text, wrap_memory([render_seed_block(seed)])))) - base_tokens
        )

    if include_seed and not seed_in_base and not seed_required:
        blocks = [render_seed_block(seed)]
        for rank, chosen in enumerate(selected, start=1):
            blocks.append(render_evidence_block(chosen.message, rank))
        used_seed, deficit_seed = _fit(base_text, blocks, counter, window, output_reserve)
        if deficit_seed <= 0:
            seed_included = True
            seed_source = "memory_optional"
            seed_tokens = (
                int(counter.count(_join(base_text, wrap_memory([render_seed_block(seed)]))))
                - base_tokens
            )
        else:
            seed_source = "omitted"
            seed_tokens = 0

    blocks: list[str] = []
    if seed_included:
        blocks.append(render_seed_block(seed))
    for rank, chosen in enumerate(selected, start=1):
        blocks.append(render_evidence_block(chosen.message, rank))
    memory_text = wrap_memory(blocks)
    final_text = _join(base_text, memory_text)
    used_tokens = int(counter.count(final_text))
    total_with_reserve = used_tokens + int(output_reserve)
    if total_with_reserve > window:
        raise AssertionError("Contexto final excede a janela; seleção inconsistente")
    remaining_tokens = window - output_reserve - used_tokens

    decisions: list[CandidateDecision] = []
    for candidate in ordered:
        reason, detail = reasons.get(
            candidate.message.id,
            (
                "outside_radius" if not candidate.within_radius else "not_evaluated",
                None,
            ),
        )
        rank = None
        if candidate.message.id in selected_ids:
            rank = next(
                i for i, chosen in enumerate(selected, start=1)
                if chosen.message.id == candidate.message.id
            )
        decisions.append(
            _decision_from(
                candidate,
                reason,
                detail,
                rank=rank,
                tokens=int(counter.count(render_evidence_block(candidate.message, 1))),
            )
        )

    diagnostics = BudgetDiagnostics(
        counter_name=counter.name,
        counter_simulated=bool(counter.simulated),
        window=int(window),
        output_reserve=int(output_reserve),
        base_tokens=base_tokens,
        capacity=capacity,
        memory_budget=memory_budget,
        memory_tokens=used_tokens - base_tokens,
        used_tokens=used_tokens,
        remaining_tokens=remaining_tokens,
        total_with_reserve=total_with_reserve,
        exceeds_capacity=False,
        status="ok",
        base_overflow_tokens=0,
        seed_included=seed_included,
        seed_in_base=seed_in_base,
        seed_source=seed_source,
        seed_tokens=seed_tokens,
        selected_ids=tuple(chosen.message.id for chosen in selected),
        decisions=tuple(decisions),
    )
    return BudgetResult(context_text=final_text, memory_text=memory_text, diagnostics=diagnostics)
