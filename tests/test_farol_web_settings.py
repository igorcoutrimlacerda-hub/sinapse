"""Fluxo HTTP do painel DeepSeek com cliente simulado (sem rede)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from sinapse.deepseek import ChatCompletionResponse
from sinapse.web_server import create_server


class RecordingClient:
    """Registra qual chave e modelo chegaram ao cliente."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.requests: list[dict] = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(dict(request_body))
        return ChatCompletionResponse(
            content="resposta simulada",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={},
        )


class FailingClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def complete(self, request_body, timeout: float = 60.0):
        raise RuntimeError(f"falha simulada com a chave {self.api_key}")


class FarolWebSettingsTest(unittest.TestCase):
    def start(self, base: Path, *, environ=None, failing=False):
        made: list = []

        def factory(key: str):
            client = FailingClient(key) if failing else RecordingClient(key)
            made.append(client)
            return client

        server = create_server(
            base / "projects",
            port=0,
            settings_path=base / "settings.json",
            client_factory=factory,
            environ={} if environ is None else environ,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server, made

    def request(self, server, method: str, path: str, payload=None):
        base = f"http://127.0.0.1:{server.server_address[1]}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_save_replace_remove_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            server, made = self.start(base)

            status, payload = self.request(server, "GET", "/api/settings")
            self.assertEqual(status, 200)
            self.assertEqual(payload["mode"], "local")
            self.assertEqual(payload["source"], "none")
            self.assertFalse(payload["configured"])

            key_one = "sk-ficticia-painel-um-0001"
            _, saved = self.request(
                server,
                "POST",
                "/api/settings/key",
                {"api_key": key_one, "model": "deepseek-flash"},
            )
            self.assertEqual(saved["source"], "panel")
            self.assertEqual(saved["mode"], "deepseek")
            self.assertNotIn(key_one, json.dumps(saved))

            _, tested = self.request(server, "POST", "/api/settings/test", {})
            self.assertTrue(tested["ok"])
            self.assertEqual(tested["source"], "panel")
            self.assertEqual(made[-1].api_key, key_one)
            self.assertEqual(made[-1].requests[0]["model"], "deepseek-flash")
            self.assertNotIn(key_one, json.dumps(tested))

            _, project = self.request(
                server, "POST", "/api/projects", {"name": "Teste DeepSeek"}
            )
            project_id = project["project"]["id"]
            _, chat = self.request(
                server,
                "POST",
                f"/api/projects/{project_id}/messages",
                {"text": "Onde paramos?", "deepseek": True},
            )
            self.assertEqual(chat["source"], "deepseek")
            self.assertEqual(made[-1].api_key, key_one)
            self.assertEqual(chat["assistant"]["text"], "resposta simulada")

            key_two = "sk-ficticia-painel-dois-0002"
            _, replaced = self.request(
                server,
                "POST",
                "/api/settings/key",
                {"api_key": key_two, "model": "deepseek-flash"},
            )
            self.assertEqual(replaced["source"], "panel")
            self.assertNotIn(key_two, json.dumps(replaced))
            _, chat_two = self.request(
                server,
                "POST",
                f"/api/projects/{project_id}/messages",
                {"text": "Segunda mensagem", "deepseek": True},
            )
            self.assertEqual(chat_two["source"], "deepseek")
            self.assertEqual(made[-1].api_key, key_two)

            # Reinicia o servidor com o mesmo arquivo de configuração.
            server.shutdown()
            server.server_close()
            restarted, restarted_made = self.start(base)
            _, persisted = self.request(restarted, "GET", "/api/settings")
            self.assertEqual(persisted["source"], "panel")
            self.assertTrue(persisted["has_saved_key"])
            self.assertNotIn(key_two, json.dumps(persisted))
            _, chat_after = self.request(
                restarted,
                "POST",
                f"/api/projects/{project_id}/messages",
                {"text": "Depois de reiniciar", "deepseek": True},
            )
            self.assertEqual(chat_after["source"], "deepseek")
            self.assertEqual(restarted_made[-1].api_key, key_two)

            # Remoção: sem chave salva e sem ambiente, volta ao modo local.
            _, removed = self.request(restarted, "POST", "/api/settings/remove", {})
            self.assertEqual(removed["mode"], "local")
            self.assertEqual(removed["source"], "none")
            self.assertFalse(removed["has_saved_key"])
            _, no_key = self.request(restarted, "POST", "/api/settings/test", {})
            self.assertFalse(no_key["ok"])
            self.assertIn("Nenhuma chave", no_key["error"])

            # Nada da chave foi para a memória do projeto.
            for path in (base / "projects").glob("*.json"):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn(key_one, content)
                self.assertNotIn(key_two, content)
            for client in made:
                self.assertIn(client.api_key, {key_one, key_two})

    def test_environment_fallback_after_remove(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env_key = "sk-ficticia-ambiente-0003"
            server, made = self.start(base, environ={"DEEPSEEK_API_KEY": env_key})

            _, status = self.request(server, "GET", "/api/settings")
            self.assertEqual(status["source"], "environment")
            self.assertEqual(status["mode"], "deepseek")
            self.assertTrue(status["environment_available"])
            self.assertFalse(status["has_saved_key"])
            self.assertNotIn(env_key, json.dumps(status))

            _, saved = self.request(
                server,
                "POST",
                "/api/settings/key",
                {"api_key": "sk-ficticia-painel-0004"},
            )
            self.assertEqual(saved["source"], "panel")
            _, removed = self.request(server, "POST", "/api/settings/remove", {})
            self.assertEqual(removed["source"], "environment")
            self.assertTrue(removed["configured"])

            _, tested = self.request(server, "POST", "/api/settings/test", {})
            self.assertTrue(tested["ok"])
            self.assertEqual(tested["source"], "environment")
            self.assertEqual(made[-1].api_key, env_key)

    def test_errors_are_redacted_and_logs_do_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            key = "sk-ficticia-segredo-0005"
            server, _ = self.start(base, failing=True)
            captured = io.StringIO()
            with contextlib.redirect_stderr(captured):
                _, saved = self.request(
                    server,
                    "POST",
                    "/api/settings/key",
                    {"api_key": key},
                )
                _, failed = self.request(server, "POST", "/api/settings/test", {})
                _, chat = self.request(
                    server,
                    "POST",
                    "/api/projects",
                    {"name": "Erro simulado"},
                )
                project_id = chat["project"]["id"]
                _, chat = self.request(
                    server,
                    "POST",
                    f"/api/projects/{project_id}/messages",
                    {"text": "Teste de erro", "deepseek": True},
                )
            self.assertNotIn(key, json.dumps(saved))
            self.assertFalse(failed["ok"])
            self.assertIn("[chave redigida]", failed["error"])
            self.assertNotIn(key, failed["error"])
            self.assertEqual(chat["source"], "local")
            self.assertIn("[chave redigida]", chat["warning"] or "")
            self.assertNotIn(key, chat["warning"] or "")
            self.assertNotIn(key, captured.getvalue())
            for path in (base / "projects").glob("*.json"):
                self.assertNotIn(key, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
