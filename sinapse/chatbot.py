"""Orquestração: evocação Sinapse → orçamento de chat → API DeepSeek."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .budget import RecentTurn, SimulatedCounter
from .chat import (
    ChatBudgetResult,
    ChatTokenCounter,
    EstimatedChatCounter,
    select_chat_context,
)
from .config import load_config
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .memory import recall
from .models import Message, RecallResult


@dataclass(frozen=True)
class ChatTurn:
    recall: RecallResult
    budget: ChatBudgetResult
    request_body: Mapping[str, Any]
    response: ChatCompletionResponse

    def as_dict(self) -> dict:
        return {
            "recall": self.recall.as_dict(),
            "budget": self.budget.as_dict(),
            "request": dict(self.request_body),
            "response": self.response.as_dict(),
        }


class SinapseChatbot:
    """Chatbot com memória: recupera, orça e envia a requisição final."""

    def __init__(
        self,
        messages: Sequence[Message],
        config: Mapping[str, Any] | None = None,
        client: ChatTransport | None = None,
        counter: ChatTokenCounter | None = None,
        settings: DeepSeekSettings | None = None,
    ) -> None:
        self.messages = list(messages)
        self.config = dict(config or load_config())
        self.settings = settings or load_deepseek_settings(self.config)
        if client is None:
            raise ValueError("Um cliente de chat é obrigatório (real ou simulado)")
        self.client = client
        self.counter = counter or EstimatedChatCounter(SimulatedCounter())

    def ask(
        self,
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
        base_message_ids: Sequence[str] = (),
        include_seed: bool = True,
        conversation_id: str | None = None,
    ) -> ChatTurn:
        turns = tuple(
            turn if isinstance(turn, RecentTurn) else RecentTurn(text=str(turn))
            for turn in recent_turns
        )
        recall_result = recall(
            self.messages,
            query,
            [turn.text for turn in turns],
            self.config,
            seed_id=seed_id,
            focus=focus,
            radius=radius,
            conversation_id=conversation_id,
        )
        budget = select_chat_context(
            instructions=str(self.config.get("instructions", "")),
            recent_turns=turns,
            current_turn=query,
            seed=recall_result.seed,
            focus=recall_result.focus,
            candidates=recall_result.candidates,
            window=int(window if window is not None else self.settings.context_window),
            output_reserve=int(
                output_reserve if output_reserve is not None else self.settings.output_reserve
            ),
            counter=self.counter,
            top_k=int(top_k if top_k is not None else self.config.get("top_k", 3)),
            threshold=float(
                threshold
                if threshold is not None
                else self.config.get("attention_threshold", 0.05)
            ),
            include_seed=include_seed,
            explicit_message_ids=base_message_ids,
            min_gain=float(self.config.get("min_marginal_gain", 0.05)),
            min_novelty=float(self.config.get("min_marginal_novelty", 0.55)),
        )
        request_body = build_request_body(budget.messages, self.settings)
        response = self.client.complete(request_body, timeout=self.settings.timeout)
        return ChatTurn(
            recall=recall_result,
            budget=budget,
            request_body=request_body,
            response=response,
        )
