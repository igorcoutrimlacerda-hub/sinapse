"""Teste do experimento de instrução com transporte falso."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.deepseek import ChatCompletionResponse, load_deepseek_settings
from sinapse.instruction_experiment import (
    export_instruction_experiment,
    run_instruction_experiment,
)


class FakeInstructionClient:
    name = "fake-instruction"
    simulated = False

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(request_body)
        return ChatCompletionResponse(
            content="Não tenho informação suficiente.",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            raw={"fake": True},
        )


class InstructionExperimentTest(unittest.TestCase):
    def test_invariants_calls_and_blind_export(self) -> None:
        config = load_config()
        settings = load_deepseek_settings(config)
        client = FakeInstructionClient()
        result = run_instruction_experiment(config, settings, client)
        self.assertEqual(result["calls"], 20)
        self.assertEqual(result["pairs"], 10)
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual(len(result["blind_review"]), 10)
        self.assertEqual(len(client.requests), 20)
        self.assertEqual(result["real_tokens"]["prompt_tokens"], 2000)
        self.assertEqual(result["real_tokens"]["completion_tokens"], 200)
        self.assertNotIn("instruction", result["blind_review"][0])
        self.assertNotIn("condition", result["blind_review"][0])
        for index in range(0, len(client.requests), 2):
            first = client.requests[index]
            second = client.requests[index + 1]
            self.assertEqual(first["messages"][1:], second["messages"][1:])
            self.assertNotEqual(
                first["messages"][0]["content"],
                second["messages"][0]["content"],
            )
            for key in ("model", "max_tokens", "temperature", "thinking", "stream"):
                self.assertEqual(first[key], second[key])
        with tempfile.TemporaryDirectory() as directory:
            paths = export_instruction_experiment(result, directory)
            self.assertTrue(Path(paths["results"]).exists())
            self.assertTrue(Path(paths["review"]).exists())
            review = json.loads(Path(paths["review"]).read_text(encoding="utf-8"))
            self.assertEqual(len(review), 10)
            self.assertTrue(
                all(
                    set(item["answers"]) == {"A", "B"}
                    and item["human_review"]["reviewed"] is False
                    for item in review
                )
            )


if __name__ == "__main__":
    unittest.main()
