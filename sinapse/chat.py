"""Orçamento aplicado ao formato de mensagens de chat.

O contexto entregue ao modelo é uma lista ``messages`` no formato da API
(``system``/``user``/``assistant``), não o texto de diagnóstico da Sinapse.
A contagem padrão é uma **estimativa** determinística: tokens de
palavras/pontuação + marcadores de papel + um overhead fixo por mensagem.
Ela nunca é apresentada como contagem real do tokenizador do DeepSeek.
Injete outro ``ChatTokenCounter`` para usar um tokenizador de verdade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol, Sequence

from .attributes import (
    ATTRIBUTE_PRICE,
    attribute_targets,
    message_supports_attribute,
    requested_attribute,
)
from .budget import (
    BudgetDiagnostics,
    CandidateDecision,
    RecentTurn,
    SimulatedCounter,
    TokenCounter,
    _decision_from,
    is_duplicate,
    render_evidence_block,
    render_seed_block,
    wrap_memory,
)
from .memory import requires_seed
from .models import Candidate, Message
from .utility import estimate_utility


class ChatTokenCounter(Protocol):
    name: str
    simulated: bool

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int: ...


@dataclass(frozen=True)
class EstimatedChatCounter:
    """Contagem estimada do payload de chat, sempre rotulada como simulada.

    ``count_messages`` conta o conteúdo, os marcadores de papel e um overhead
    fixo por mensagem. Isso aproxima o custo do template de chat, mas **não**
    é o tokenizador oficial; o uso real informado pela API é a referência.
    """

    counter: TokenCounter = field(default_factory=SimulatedCounter)
    per_message_overhead: int = 4

    @property
    def name(self) -> str:
        return f"{self.counter.name}+chat-overhead-{self.per_message_overhead}"

    @property
    def simulated(self) -> bool:
        return bool(self.counter.simulated)

    def count_messages(self, messages: Sequence[Mapping[str, str]]) -> int:
        rendered = "".join(
            f"<|{message['role']}|>\n{message['content']}\n<|end|>\n"
            for message in messages
        )
        return int(self.counter.count(rendered)) + self.per_message_overhead * len(messages)


ASSISTANT_AUTHORS = frozenset({"assistant", "bot", "model", "ai", "agente"})


def role_for_author(author: str) -> str:
    return "assistant" if str(author).strip().lower() in ASSISTANT_AUTHORS else "user"


def build_base_messages(
    instructions: str,
    recent_turns: Sequence[RecentTurn],
    current_turn: str,
) -> tuple[dict[str, str], ...]:
    """Instruções, histórico recente e turno atual, sem memória.

    Turnos com ``id`` recebem o ID no início do conteúdo para preservar a
    atribuição também no formato de chat.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": instructions.strip() or "(sem instruções)"}
    ]
    for turn in recent_turns:
        content = f"[{turn.id}] {turn.text}" if turn.id else turn.text
        messages.append({"role": role_for_author(turn.author), "content": content})
    messages.append({"role": "user", "content": current_turn})
    return tuple(messages)


def with_memory(
    base_messages: Sequence[Mapping[str, str]],
    memory_text: str,
) -> tuple[dict[str, str], ...]:
    """Anexa a memória ao primeiro ``system``; cria um se não existir."""
    messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in base_messages]
    if not memory_text:
        return tuple(messages)
    for index, message in enumerate(messages):
        if message["role"] == "system":
            messages[index] = {
                "role": "system",
                "content": message["content"] + "\n\n" + memory_text,
            }
            return tuple(messages)
    messages.insert(0, {"role": "system", "content": memory_text})
    return tuple(messages)


@dataclass(frozen=True)
class ChatBudgetResult:
    messages: tuple[dict[str, str], ...]
    memory_text: str
    diagnostics: BudgetDiagnostics

    def as_dict(self) -> dict:
        return {
            "messages": [dict(message) for message in self.messages],
            "memory_text": self.memory_text,
            "diagnostics": self.diagnostics.as_dict(),
        }


def select_chat_context(
    instructions: str,
    recent_turns: Sequence[RecentTurn],
    current_turn: str,
    seed: Message,
    focus: str,
    candidates: Sequence[Candidate],
    window: int,
    output_reserve: int,
    counter: ChatTokenCounter,
    top_k: int = 3,
    threshold: float = 0.05,
    include_seed: bool = True,
    explicit_message_ids: Sequence[str] = (),
    min_gain: float = 0.05,
    min_novelty: float = 0.55,
    use_utility: bool = True,
    redundancy_weight: float = 0.6,
) -> ChatBudgetResult:
    """Seleciona memória medindo o payload de chat completo.

    As regras de elegibilidade são as mesmas do orçamento textual: raio,
    evidência não nula no foco, atenção acima do limiar, dedup conservadora,
    IDs já presentes na base, K e encaixe exato. Distância, vetor e atenção
    ficam apenas no diagnóstico.
    """
    base_messages = build_base_messages(instructions, recent_turns, current_turn)
    base_tokens = int(counter.count_messages(base_messages))
    capacity = int(window) - int(output_reserve)
    memory_budget = max(0, capacity - base_tokens)
    base_ids = {turn.id for turn in recent_turns if turn.id}
    base_ids.update(explicit_message_ids)
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
            context_format="chat-messages",
            min_marginal_gain=float(min_gain),
        )
        return ChatBudgetResult(
            messages=base_messages, memory_text="", diagnostics=diagnostics
        )

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
            context_format="chat-messages",
            min_marginal_gain=float(min_gain),
        )
        return ChatBudgetResult(
            messages=base_messages, memory_text="", diagnostics=diagnostics
        )

    def pack(
        seed_included: bool,
        chosen: Sequence[Candidate],
        extra: Candidate | None = None,
    ) -> tuple[int, str, tuple[dict[str, str], ...]]:
        blocks: list[str] = []
        if seed_included:
            blocks.append(render_seed_block(seed))
        for rank, item in enumerate(chosen, start=1):
            blocks.append(render_evidence_block(item.message, rank))
        if extra is not None:
            blocks.append(render_evidence_block(extra.message, len(chosen) + 1))
        memory_text = wrap_memory(blocks)
        messages = with_memory(base_messages, memory_text)
        return int(counter.count_messages(messages)), memory_text, messages

    attribute = requested_attribute(current_turn)
    if attribute == ATTRIBUTE_PRICE:
        targets = attribute_targets(current_turn)
        by_id = {candidate.message.id: candidate.message for candidate in candidates}
        by_id[seed.id] = seed
        supporting = {
            candidate.message.id
            for candidate in candidates
            if message_supports_attribute(
                candidate.message, attribute, targets, by_id
            )
        }
        seed_supports = message_supports_attribute(seed, attribute, targets, by_id)
        if not supporting and not seed_supports:
            decisions = tuple(
                _decision_from(
                    candidate,
                    "no_attribute_evidence",
                    "atributo preço não registrado; tema não substitui evidência",
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
                memory_budget=memory_budget,
                memory_tokens=0,
                used_tokens=base_tokens,
                remaining_tokens=capacity - base_tokens,
                total_with_reserve=base_tokens + int(output_reserve),
                exceeds_capacity=False,
                status="no_pertinent_evidence",
                base_overflow_tokens=0,
                seed_included=False,
                seed_in_base=seed_in_base,
                seed_source="base" if seed_in_base else "omitted",
                seed_tokens=0,
                selected_ids=(),
                decisions=decisions,
                context_format="chat-messages",
                min_marginal_gain=float(min_gain),
            )
            return ChatBudgetResult(
                messages=base_messages, memory_text="", diagnostics=diagnostics
            )
        if supporting:
            candidates = [
                replace(
                    candidate,
                    evidence={**candidate.evidence, focus: 1.0},
                    attention=max(candidate.attention, candidate.association),
                )
                for candidate in candidates
                if candidate.message.id in supporting
            ]
        else:
            candidates = []

    ordered = sorted(
        candidates,
        key=lambda c: (-c.attention, c.distance, c.message.position, c.message.id),
    )
    reasons: dict[str, tuple[str, str | None]] = {}
    gains: dict[str, float] = {}
    utility_notes: dict[str, str] = {}
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    seed_required = False
    stopped_early = False
    stop_reason: str | None = None
    recent_texts = [turn.text for turn in recent_turns]

    def usable(candidate: Candidate) -> bool:
        return (
            candidate.within_radius
            and candidate.evidence.get(focus) is not None
            and candidate.attention > threshold
            and candidate.message.id not in base_ids
            and len(selected) < top_k
        )

    def utility_for(candidate: Candidate):
        return estimate_utility(
            candidate.message,
            candidate.evidence.get(focus),
            candidate.association,
            recent_texts,
            [chosen.message for chosen in selected],
            redundancy_weight=redundancy_weight,
        )

    pending = list(ordered)
    while pending:
        candidate = pending.pop(0)
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

        utility = utility_for(candidate)
        gains[message.id] = utility.gain
        utility_notes[message.id] = utility.reason
        if use_utility and (
            utility.same_assertion
            or utility.novelty < min_novelty
            or utility.gain < min_gain
        ):
            if utility.same_assertion:
                reasons[message.id] = (
                    "same_assertion_in_context",
                    f"ganho {utility.gain:.3f}; {utility.reason}",
                )
            elif utility.novelty < min_novelty:
                reasons[message.id] = (
                    "below_marginal_gain",
                    f"ganho {utility.gain:.3f}; novidade {utility.novelty:.3f} "
                    f"< {min_novelty:.3f}; {utility.reason}",
                )
            else:
                reasons[message.id] = (
                    "below_marginal_gain",
                    f"ganho {utility.gain:.3f} < limiar {min_gain:.3f}; "
                    f"{utility.reason}",
                )
            remaining_strong = False
            for rest in pending:
                if usable(rest):
                    rest_utility = utility_for(rest)
                    gains.setdefault(rest.message.id, rest_utility.gain)
                    utility_notes.setdefault(rest.message.id, rest_utility.reason)
                    if (
                        not rest_utility.same_assertion
                        and rest_utility.novelty >= min_novelty
                        and rest_utility.gain >= min_gain
                    ):
                        remaining_strong = True
                        break
            if not remaining_strong:
                stopped_early = True
                stop_reason = (
                    "candidatos restantes com novidade/ganho abaixo do limiar; "
                    "seleção encerrada com espaço sobrando"
                )
                for rest in pending:
                    if rest.message.id not in reasons:
                        reasons[rest.message.id] = (
                            "below_marginal_gain",
                            "seleção encerrada com espaço sobrando",
                        )
                pending.clear()
            continue

        needs_seed = requires_seed(message, seed) and not seed_in_base
        if needs_seed and not include_seed:
            reasons[message.id] = (
                "requires_seed",
                "resposta depende da semente e include_seed=False",
            )
            continue
        include_seed_now = seed_required or needs_seed
        used, _, _ = pack(include_seed_now, selected, candidate)
        deficit = used + int(output_reserve) - int(window)
        if deficit <= 0:
            selected.append(candidate)
            selected_ids.add(message.id)
            if needs_seed:
                seed_required = True
            reasons[message.id] = (
                "selected",
                f"ganho {utility.gain:.3f}; {utility.reason}; "
                f"posição {len(selected)}; requisição com {used} tokens estimados",
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
        seed_tokens = pack(True, ())[0] - base_tokens

    if include_seed and not seed_in_base and not seed_required:
        used_seed, _, _ = pack(True, selected)
        if used_seed + int(output_reserve) <= int(window):
            seed_included = True
            seed_source = "memory_optional"
            seed_tokens = pack(True, ())[0] - base_tokens
        else:
            seed_source = "omitted"
            seed_tokens = 0

    used_tokens, memory_text, final_messages = pack(seed_included, selected)
    total_with_reserve = used_tokens + int(output_reserve)
    if total_with_reserve > int(window):
        raise AssertionError("Contexto final excede a janela; seleção inconsistente")
    remaining_tokens = int(window) - int(output_reserve) - used_tokens

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
        token_cost = (
            pack(False, (), candidate)[0] - base_tokens
            if candidate.within_radius
            else 0
        )
        decisions.append(
            _decision_from(
                candidate,
                reason,
                detail,
                rank=rank,
                tokens=token_cost,
                marginal_gain=gains.get(candidate.message.id),
                selection_reason=utility_notes.get(candidate.message.id),
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
        context_format="chat-messages",
        min_marginal_gain=float(min_gain),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )
    return ChatBudgetResult(
        messages=final_messages, memory_text=memory_text, diagnostics=diagnostics
    )
