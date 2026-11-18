"""Sinapse: memória inspecionável para chatbots."""

from .models import DIMENSIONS, Candidate, Group, Message, RecallResult
from .memory import recall
from .budget import (
    BaseInput,
    CharCounter,
    RecentTurn,
    SimulatedCounter,
    select_context,
)
from .grouping import build_groups
from .chat import (
    ChatBudgetResult,
    EstimatedChatCounter,
    build_base_messages,
    select_chat_context,
)
from .chatbot import ChatTurn, SinapseChatbot
from .deepseek import (
    ChatCompletionResponse,
    DeepSeekClient,
    DeepSeekSettings,
    SimulatedDeepSeekClient,
    build_request_body,
    load_deepseek_settings,
)
from .themes import ThemeReport, analyze_turn, theme_affinity
from .utility import estimate_utility
from .sensitivity import (
    DerivedSummary,
    Preference,
    SensitivityPolicy,
    SensitivityReport,
)
from .whiteboard import AgentTurn, ConversationWhiteboard, MemoryAgent, send_agent_turn

__all__ = [
    "DIMENSIONS",
    "Candidate",
    "Group",
    "Message",
    "RecallResult",
    "BaseInput",
    "RecentTurn",
    "CharCounter",
    "SimulatedCounter",
    "select_context",
    "recall",
    "build_groups",
    "ChatBudgetResult",
    "EstimatedChatCounter",
    "build_base_messages",
    "select_chat_context",
    "ChatTurn",
    "SinapseChatbot",
    "ChatCompletionResponse",
    "DeepSeekClient",
    "DeepSeekSettings",
    "SimulatedDeepSeekClient",
    "build_request_body",
    "load_deepseek_settings",
    "ThemeReport",
    "analyze_turn",
    "theme_affinity",
    "estimate_utility",
    "DerivedSummary",
    "Preference",
    "SensitivityPolicy",
    "SensitivityReport",
    "AgentTurn",
    "ConversationWhiteboard",
    "MemoryAgent",
    "send_agent_turn",
]
