"""Quadro antecipatório: hipótese, validade, reserva e envio do principal."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.deepseek import ChatCompletionResponse, load_deepseek_settings
from sinapse.project_copilot import ProjectStore, chat_with_deepseek


class FakeChatClient:
    name = "fake-chat"
    simulated = False

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(request_body)
        return ChatCompletionResponse(
            content="Resposta simulada.",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            raw={"fake": True},
        )


def config_with_anticipation() -> dict:
    config = load_config()
    config["anticipation"] = {
        "enabled": True,
        "max_rounds": 1,
        "max_candidates": 8,
        "max_time_ms": 150,
        "valid_turns": 1,
    }
    return config


def seed_project(store: ProjectStore) -> dict:
    project = store.create("Sinapse")
    for text in (
        "Comecei a Sinapse com o preset Pigments no bass.",
        "Falta testar a ponte em 96 kHz.",
        "Se a CPU ficar abaixo de 30%, o próximo passo é exportar; senão, reduzo o oversampling.",
        "As configurações de exportação são 48 kHz e 24 bits.",
        "A CPU ficou em 27%.",
    ):
        store.add_message(store.load(), text)
    return store.load()


class AnticipationTest(unittest.TestCase):
    def test_27_percent_reserves_export_settings_and_promotes_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            config = config_with_anticipation()
            seed_project(store)
            thread = store.start_anticipation("sinapse", "A CPU ficou em 27%.", config)
            thread.join(timeout=2)
            project = store.load()
            board = project["anticipatory"]
            self.assertEqual(board["status"], "ready")
            self.assertIn("export", board["hypothesis"].lower())
            self.assertIn("m4", board["source_ids"])
            self.assertEqual(board["turn"], project["turn"])
            self.assertEqual(board["project_revision"], project["revision"])

            decision_before = project["state"]["decisions"][-1]["value"]
            pending_before = project["state"]["pending"][0]["status"]
            reservation = store.consume_anticipation(
                "sinapse", "Como exporto em 48 kHz?"
            )
            self.assertIsNotNone(reservation)
            project = store.load()
            store.add_message(project, "Como exporto em 48 kHz?")
            project = store.load()
            self.assertTrue(store.valid_anticipation(project, reservation))
            client = FakeChatClient()
            answer = chat_with_deepseek(
                project,
                "Como exporto em 48 kHz?",
                config,
                reserved_ids=reservation["source_ids"],
                client=client,
            )
            self.assertEqual(answer, "Resposta simulada.")
            sent = "\n".join(
                str(message["content"]) for message in client.requests[0]["messages"]
            )
            self.assertIn("configurações de exportação", sent)
            project = store.load()
            self.assertEqual(
                project["state"]["decisions"][-1]["value"], decision_before
            )
            self.assertEqual(project["state"]["pending"][0]["status"], pending_before)

    def test_subject_change_expires_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            config = config_with_anticipation()
            seed_project(store)
            thread = store.start_anticipation("sinapse", "A CPU ficou em 27%.", config)
            thread.join(timeout=2)
            self.assertIsNone(
                store.consume_anticipation("sinapse", "Por que escolhi o Pigments?")
            )
            self.assertIsNotNone(store.load()["anticipatory"].get("expired_reason"))

    def test_late_result_is_discarded_after_project_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            config = config_with_anticipation()
            seed_project(store)
            store.create("Outro")
            thread = store.start_anticipation(
                "sinapse", "A CPU ficou em 27%.", config, delay_seconds=0.2
            )
            store.select("outro")
            thread.join(timeout=2)
            board = store.load("sinapse")["anticipatory"]
            self.assertNotEqual(board["status"], "ready")

    def test_in_progress_does_not_block_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            config = config_with_anticipation()
            seed_project(store)
            thread = store.start_anticipation(
                "sinapse", "A CPU ficou em 27%.", config, delay_seconds=0.2
            )
            started = time.perf_counter()
            reservation = store.consume_anticipation(
                "sinapse", "Como exporto em 48 kHz?"
            )
            elapsed = time.perf_counter() - started
            self.assertIsNone(reservation)
            self.assertLess(elapsed, 0.1)
            store.add_message(store.load(), "Como exporto em 48 kHz?")
            thread.join(timeout=2)

    def test_restricted_source_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sigilo")
            store.add_message(store.load(), "Comecei com o preset X.")
            store.add_message(store.load(), "A configuração de exportação é segredo.")
            store.restrict("segredo")
            project = store.load()
            board = {
                "source_ids": ["m2"],
                "hypothesis": "pedir exportação",
            }
            self.assertFalse(store.valid_anticipation(project, board))

    def test_superseded_decision_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sinapse")
            store.add_message(store.load(), "Comecei com o preset Aurora no bass.")
            store.add_message(
                store.load(),
                "Na verdade, troquei para o preset Pigments porque a CPU do Aurora era alta.",
            )
            project = store.load()
            board = {"source_ids": ["m1"], "hypothesis": "continuar"}
            self.assertFalse(store.valid_anticipation(project, board))


if __name__ == "__main__":
    unittest.main()
