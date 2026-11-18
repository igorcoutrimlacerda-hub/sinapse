"""Configuração global do DeepSeek no painel do Farol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.deepseek import ChatCompletionResponse, load_deepseek_settings
from sinapse.farol_settings import FarolSettings


class FakeClient:
    simulated = False

    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.requests = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(request_body)
        if self.error:
            raise RuntimeError(self.error)
        return ChatCompletionResponse(
            content="ok",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={},
        )


class FarolSettingsTest(unittest.TestCase):
    def test_save_replace_remove_and_environment_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = FarolSettings(path, environ={})
            self.assertEqual(settings.status()["mode"], "local")
            status = settings.save_key("sk-panel", "deepseek-flash")
            self.assertTrue(status["configured"])
            self.assertEqual(status["source"], "panel")
            self.assertNotIn("sk-panel", json.dumps(status))
            self.assertTrue(path.exists())
            reloaded = FarolSettings(path, environ={})
            self.assertEqual(reloaded.active_key(), ("sk-panel", "panel"))
            reloaded.save_key("sk-replaced")
            self.assertEqual(reloaded.active_key()[0], "sk-replaced")
            removed = reloaded.remove_key()
            self.assertEqual(removed["mode"], "local")
            self.assertFalse(removed["configured"])

            with_env = FarolSettings(path, environ={"DEEPSEEK_API_KEY": "sk-env"})
            self.assertEqual(with_env.active_key(), ("sk-env", "environment"))
            with_env.save_key("sk-panel")
            with_env.remove_key()
            self.assertEqual(with_env.status()["source"], "environment")
            self.assertTrue(with_env.status()["configured"])

    def test_status_never_contains_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = FarolSettings(Path(directory) / "s.json", environ={})
            settings.save_key("sk-secret-value")
            serialized = json.dumps(settings.status())
            self.assertNotIn("sk-secret-value", serialized)
            self.assertEqual(settings.status()["model"], "deepseek-flash")
            self.assertEqual(
                settings.status()["model_version"], "DeepSeek-V4.1-Flash-1017"
            )

    def test_test_connection_uses_active_key_and_redacts_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = FarolSettings(Path(directory) / "s.json", environ={})
            settings.save_key("sk-hidden")
            client = FakeClient()
            result = settings.test_connection(client)
            self.assertTrue(result["ok"])
            self.assertEqual(result["source"], "panel")
            self.assertEqual(client.requests[0]["model"], "deepseek-flash")
            failing = FakeClient(error="falha com sk-hidden")
            result = settings.test_connection(failing)
            self.assertFalse(result["ok"])
            self.assertNotIn("sk-hidden", result["error"])
            self.assertIn("[chave redigida]", result["error"])

    def test_no_key_does_not_call_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = FarolSettings(Path(directory) / "s.json", environ={})
            client = FakeClient()
            result = settings.test_connection(client)
            self.assertFalse(result["ok"])
            self.assertEqual(result["source"], "none")
            self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
