"""Testes da integração DeepSeek: configuração, HTTP, orçamento e cliente simulado."""

from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from sinapse.budget import RecentTurn
from sinapse.chat import EstimatedChatCounter, build_base_messages
from sinapse.chatbot import SinapseChatbot
from sinapse.config import load_config
from sinapse.deepseek import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekSettings,
    MissingApiKey,
    SimulatedDeepSeekClient,
    build_request_body,
    load_deepseek_settings,
)
from sinapse.models import load_messages
from sinapse.whiteboard import MemoryAgent, send_agent_turn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_conversation.json"
CONFIG = ROOT / "data" / "config.json"

QUERY = "Por que eu escolhi o Serum 2?"
RECENT = ["Estávamos falando do Serum 2 e do bass."]


class SettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)

    def test_defaults_follow_official_documentation(self) -> None:
        settings = load_deepseek_settings(self.config)
        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.model_version, "DeepSeek-V4.1-Flash-1017")
        self.assertEqual(settings.context_window, DEFAULT_CONTEXT_WINDOW)
        self.assertEqual(settings.context_window, 1_000_000)
        self.assertEqual(settings.output_reserve, 4_096)
        self.assertEqual(settings.max_tokens, 4_096)
        self.assertEqual(settings.thinking, "disabled")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")

    def test_official_model_ids_only(self) -> None:
        with self.assertRaises(ValueError):
            load_deepseek_settings({"deepseek": {"model": "deepseek-v4-flash"}})
        settings = load_deepseek_settings({"deepseek": {"model": "deepseek-v4-pro"}})
        self.assertEqual(settings.model_version, "DeepSeek-V4-Pro-0813")

    def test_output_limit_is_384k(self) -> None:
        with self.assertRaises(ValueError):
            DeepSeekSettings(max_tokens=MAX_OUTPUT_TOKENS + 1).validate()
        DeepSeekSettings(
            max_tokens=MAX_OUTPUT_TOKENS, output_reserve=MAX_OUTPUT_TOKENS
        ).validate()

    def test_max_tokens_cannot_exceed_reserve(self) -> None:
        with self.assertRaises(ValueError):
            DeepSeekSettings(output_reserve=1000, max_tokens=2000).validate()

    def test_api_key_comes_from_environment(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            client = DeepSeekClient()
        self.assertEqual(client.api_key, "sk-test")

    def test_missing_api_key_is_explicit(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "   "}):
            with self.assertRaises(MissingApiKey):
                DeepSeekClient()


class ClientTests(unittest.TestCase):
    def test_http_request_uses_model_key_and_parses_response(self) -> None:
        captured: dict = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return json.dumps(
                {
                    "id": "chat-1",
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Você comprou a licença antes.",
                                "reasoning_content": None,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 10,
                        "total_tokens": 130,
                    },
                }
            ).encode("utf-8")

        client = DeepSeekClient(api_key="sk-test", opener=opener)
        settings = DeepSeekSettings()
        body = build_request_body(
            [
                {"role": "system", "content": "instruções"},
                {"role": "user", "content": "oi"},
            ],
            settings,
        )
        response = client.complete(body, timeout=12)

        self.assertEqual(
            captured["url"], "https://api.deepseek.com/chat/completions"
        )
        self.assertEqual(captured["authorization"], "Bearer sk-test")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["body"]["model"], "deepseek-flash")
        self.assertEqual(captured["body"]["max_tokens"], 4096)
        self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})
        self.assertFalse(captured["body"]["stream"])
        self.assertEqual(response.content, "Você comprou a licença antes.")
        self.assertEqual(response.usage["prompt_tokens"], 120)
        self.assertEqual(response.finish_reason, "stop")

    def test_http_error_is_reported_with_body(self) -> None:
        def opener(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"error":"invalid api key"}'),
            )

        client = DeepSeekClient(api_key="sk-test", opener=opener)
        with self.assertRaises(DeepSeekAPIError) as context:
            client.complete({"model": "deepseek-flash"}, timeout=5)
        self.assertEqual(context.exception.status, 401)
        self.assertIn("invalid api key", context.exception.body)


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages = load_messages(DATA)
        cls.config = load_config(CONFIG)

    def make_bot(self, reply: str = "A licença já estava comprada."):
        client = SimulatedDeepSeekClient(reply=reply)
        return SinapseChatbot(self.messages, self.config, client), client

    def test_simulated_client_captures_final_request_with_memory(self) -> None:
        bot, client = self.make_bot()
        turn = bot.ask(QUERY, RECENT)

        self.assertEqual(len(client.requests), 1)
        body = client.requests[-1]
        self.assertEqual(body, turn.request_body)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        system = body["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("id=m1", system["content"])
        self.assertIn("id=m2", system["content"])
        self.assertIn("licença", system["content"])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": QUERY})
        self.assertEqual(turn.response.content, "A licença já estava comprada.")
        self.assertEqual(turn.budget.diagnostics.selected_ids, ("m2",))

    def test_budget_is_applied_to_message_payload(self) -> None:
        bot, _ = self.make_bot()
        turn = bot.ask(QUERY, RECENT)
        d = turn.budget.diagnostics
        counter = EstimatedChatCounter()
        base = build_base_messages(
            str(self.config.get("instructions", "")),
            tuple(RecentTurn(text=text, author="user") for text in RECENT),
            QUERY,
        )
        self.assertEqual(d.base_tokens, counter.count_messages(base))
        self.assertEqual(d.used_tokens, counter.count_messages(turn.budget.messages))
        self.assertEqual(d.used_tokens - d.base_tokens, d.memory_tokens)
        self.assertLessEqual(d.total_with_reserve, d.window)
        self.assertEqual(d.remaining_tokens, d.window - d.output_reserve - d.used_tokens)
        self.assertEqual(d.context_format, "chat-messages")

    def test_tight_budget_sends_no_memory_but_keeps_diagnosis(self) -> None:
        bot, client = self.make_bot()
        base = build_base_messages(
            str(self.config.get("instructions", "")),
            tuple(RecentTurn(text=text, author="user") for text in RECENT),
            QUERY,
        )
        base_tokens = EstimatedChatCounter().count_messages(base)
        turn = bot.ask(
            QUERY,
            RECENT,
            window=base_tokens + 4096,
            output_reserve=4096,
            include_seed=False,
        )
        d = turn.budget.diagnostics
        self.assertEqual(d.memory_budget, 0)
        self.assertEqual(d.status, "no_budget")
        self.assertEqual(d.selected_ids, ())
        self.assertEqual(d.used_tokens, base_tokens)
        system = client.requests[-1]["messages"][0]
        self.assertNotIn("MEMÓRIA", system["content"])
        self.assertTrue(system["content"].startswith("Você é um assistente"))

    def test_simulated_count_is_identified(self) -> None:
        bot, _ = self.make_bot()
        turn = bot.ask(QUERY, RECENT)
        d = turn.budget.diagnostics
        self.assertTrue(d.counter_simulated)
        self.assertIn("chat-overhead", d.counter_name)
        self.assertEqual(d.counter_name, "simulated-words-punct-v1+chat-overhead-4")

    def test_agent_turn_uses_same_final_request(self) -> None:
        agent = MemoryAgent(self.messages, self.config)
        turn = agent.before_response("c1", QUERY, RECENT)
        client = SimulatedDeepSeekClient(reply="ok")
        settings = load_deepseek_settings(self.config)
        body, response = send_agent_turn(turn, client, settings)
        self.assertEqual(body, client.requests[-1])
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertIn("id=m2", body["messages"][0]["content"])
        self.assertEqual(response.content, "ok")


if __name__ == "__main__":
    unittest.main()
