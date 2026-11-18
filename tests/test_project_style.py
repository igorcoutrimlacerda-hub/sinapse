"""Estilo de conversa do Farol no modo local e no prompt canônico."""

from __future__ import annotations

import json
import tempfile
import unittest

from sinapse.config import load_config
from sinapse.deepseek import ChatCompletionResponse
from sinapse.project_copilot import (
    FAROL_CANONICAL_PROMPT,
    ProjectStore,
    chat_with_deepseek,
    local_answer,
)


class FakeClient:
    simulated = False
    name = "fake-style"

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(request_body)
        return ChatCompletionResponse(
            content="Boa — 27% fica abaixo de 30%; o caminho combinado é exportar.",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={"prompt_tokens": 80, "completion_tokens": 15},
            raw={"fake": True},
        )


def seed(store: ProjectStore) -> dict:
    store.create("Sinapse")
    for text in (
        "Comecei a Sinapse com o preset Pigments no bass porque a CPU do Aurora ficava alta.",
        "O teste do refrão ficou concluído.",
        "Falta testar a ponte em 96 kHz.",
        "Se a CPU ficar abaixo de 30%, o próximo passo é exportar; senão, reduzo o oversampling.",
    ):
        store.add_message(store.load(), text)
    return store.load()


class ProjectStyleTest(unittest.TestCase):
    def test_casual_answer_does_not_dump_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            answer = local_answer(store, project, "Oi, hoje estou mexendo na faixa Norte.")
            self.assertIn("Legal", answer)
            self.assertNotIn("Decisão vigente:", answer)
            self.assertNotIn("Checklist:", answer)

    def test_idea_is_proposal_and_does_not_change_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            before = json.dumps(project["state"], sort_keys=True, ensure_ascii=False)
            store.add_message(
                store.load(),
                "Estou pensando em testar o Pigments na introdução, mas talvez seja exagero.",
            )
            project = store.load()
            answer = local_answer(
                store,
                project,
                "Estou pensando em testar o Pigments na introdução, mas talvez seja exagero.",
            )
            after = json.dumps(project["state"], sort_keys=True, ensure_ascii=False)
            self.assertIn("proposta", answer)
            self.assertEqual(before, after)

    def test_cpu_result_uses_condition_and_keeps_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            before = json.dumps(project["state"], sort_keys=True, ensure_ascii=False)
            store.add_message(store.load(), "A CPU ficou em 27%.")
            project = store.load()
            answer = local_answer(store, project, "A CPU ficou em 27%.")
            after = json.dumps(project["state"], sort_keys=True, ensure_ascii=False)
            self.assertIn("27%", answer)
            self.assertIn("abaixo de 30%", answer)
            self.assertIn("exportar", answer)
            self.assertEqual(before, after)

    def test_resume_keeps_four_fields_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            answer = local_answer(store, project, "Onde paramos?")
            for label in ("Tarefa atual", "Decisão técnica", "Motivo", "Último teste observado", "Próximo passo"):
                self.assertIn(label, answer)
            self.assertIn("[m1]", answer)
            self.assertIn("[m3]", answer)

    def test_casual_deepseek_does_not_include_state_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            client = FakeClient()
            chat_with_deepseek(
                project,
                "Oi, hoje estou mexendo na faixa Norte.",
                load_config(),
                client=client,
                store=store,
            )
            system = client.requests[0]["messages"][0]["content"]
            self.assertIn("FAROL_CANONICAL v3", system)
            self.assertNotIn("ESTADO PERSISTIDO", system)

    def test_deepseek_receives_canonical_style_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            project = seed(store)
            client = FakeClient()
            answer = chat_with_deepseek(
                project,
                "Onde paramos?",
                load_config(),
                client=client,
                store=store,
            )
            self.assertIn("27%", answer)
            system = client.requests[0]["messages"][0]["content"]
            self.assertIn("FAROL_CANONICAL v3", system)
            self.assertIn("Português do Brasil natural", system)
            self.assertIn("propostas", system)
            self.assertNotIn("Registrei a mensagem", system)


if __name__ == "__main__":
    unittest.main()
