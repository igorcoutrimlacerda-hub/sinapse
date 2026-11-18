"""Execução pareada real com transporte falso: valida chamadas e exportação."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.deepseek import ChatCompletionResponse, load_deepseek_settings
from sinapse.resumption_real import export_real_pairs, run_real_pairs


class FakeRealClient:
    name = "fake-real"
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


class RealPairingTest(unittest.TestCase):
    def test_real_pairing_calls_and_export(self) -> None:
        config = load_config()
        settings = load_deepseek_settings(config)
        client = FakeRealClient()
        result = run_real_pairs(
            "benchmark_out_resumption_round3/real_comparison.json",
            config,
            settings,
            client,
        )
        self.assertEqual(result["calls"], 20)
        self.assertEqual(result["pairs"], 10)
        self.assertEqual(len(result["rows"]), 20)
        self.assertEqual(len(result["blind_review"]), 10)
        self.assertEqual(len(client.requests), 20)
        self.assertEqual(result["real_tokens"]["prompt_tokens"], 2000)
        self.assertEqual(result["real_tokens"]["completion_tokens"], 200)
        self.assertTrue(
            all(row["token_source"] == "api" for row in result["rows"])
        )
        self.assertTrue(all(row["error"] is None for row in result["rows"]))
        with tempfile.TemporaryDirectory() as directory:
            paths = export_real_pairs(result, directory)
            self.assertTrue(Path(paths["results"]).exists())
            self.assertTrue(Path(paths["review"]).exists())
            self.assertTrue(Path(paths["summary"]).exists())
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
