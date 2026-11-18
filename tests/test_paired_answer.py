"""Testes do benchmark pareado atual vs reduzido (modo simulado)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.paired_answer import export_paired, run_paired


class PairedAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_paired(load_config("data/config.json"))

    def test_cases_and_calls(self) -> None:
        self.assertEqual(self.result["cases"], 12)
        self.assertEqual(self.result["calls"], 24)
        self.assertEqual(
            self.result["counts"],
            {"melhorou": 0, "empatou": 12, "piorou": 0},
        )

    def test_reduction_lowers_memory_without_losing_checks(self) -> None:
        current = self.result["conditions"]["current"]
        reduced = self.result["conditions"]["reduced"]
        self.assertEqual(current["facts"], 12)
        self.assertEqual(reduced["facts"], 12)
        self.assertLessEqual(reduced["avg_memory_tokens"], current["avg_memory_tokens"])
        self.assertEqual(current["api_prompt_tokens"], 0)
        self.assertEqual(reduced["api_prompt_tokens"], 0)

    def test_pack_quantity_and_ambiguity_cases(self) -> None:
        entries = {entry["case"]: entry for entry in self.result["entries"]}
        quantity = entries["pack_quantity"]["reduced"]
        ambiguous = entries["pack_ambiguous"]["reduced"]
        self.assertTrue(quantity["checks"]["facts_ok"])
        self.assertEqual(quantity["selected"], ["m8"])
        self.assertTrue(ambiguous["checks"]["ambiguity_ok"])
        self.assertIn("m8", ambiguous["selected"])
        self.assertIn("m9", ambiguous["selected"])

    def test_token_source_is_labeled(self) -> None:
        for entry in self.result["entries"]:
            for condition in ("current", "reduced"):
                self.assertEqual(entry[condition]["token_source"], "estimativa_local")
                self.assertGreater(entry[condition]["request_tokens_estimate"], 0)

    def test_blind_review_does_not_label_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = export_paired(self.result, directory)
            review = json.loads(Path(paths["review"]).read_text(encoding="utf-8"))
            self.assertEqual(len(review), self.result["cases"])
            for entry in review:
                self.assertNotIn("current", entry)
                self.assertNotIn("reduced", entry)
                self.assertIn("answer_A", entry)
                self.assertIn("answer_B", entry)


if __name__ == "__main__":
    unittest.main()
