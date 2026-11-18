"""Testes do reprocessamento dos resultados salvos por propósito."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.purpose_report import export_human_review, reprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_out_purpose_answer_deepseek" / "purpose_answer_results.json"
REVIEW = ROOT / "benchmark_out_purpose_answer_deepseek" / "purpose_answer_review.json"


class PurposeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = reprocess(RESULTS)

    def test_overall_counts_are_separate(self) -> None:
        purpose = self.summary["overall"]["purpose"]
        self.assertEqual(purpose["rubric_approved"], 12)
        self.assertEqual(purpose["all_purposes"], 11)
        self.assertEqual(purpose["purposes_ok"], 15)
        self.assertEqual(purpose["purposes_total"], 16)
        self.assertEqual(self.summary["overall"]["reference"]["rubric_approved"], 7)

    def test_priority_case_is_approved_with_pending_part(self) -> None:
        entry = self.summary["priority_case"]
        purpose = entry["conditions"]["purpose"]
        self.assertTrue(purpose["checks"]["all_ok"])
        self.assertFalse(all(v["ok"] for v in purpose["checks"]["purposes"].values()))
        self.assertTrue(purpose["checks"]["priority_ok"])
        self.assertTrue(purpose["checks"]["purposes"]["m"]["ok"])
        self.assertFalse(purpose["checks"]["purposes"]["p"]["ok"])

    def test_groups_and_accepted_omissions(self) -> None:
        self.assertEqual(self.summary["groups"]["regression"]["purpose"]["rubric_approved"], 7)
        self.assertEqual(self.summary["groups"]["new"]["purpose"]["rubric_approved"], 5)
        reasons = {item["reason"] for item in self.summary["omissions"]}
        self.assertIn("prioridade_explicita", reasons)
        self.assertTrue(
            any(item["reason"] == "restricao_indisponivel" for item in self.summary["accepted_unavailable"])
        )

    def test_human_review_is_blind_and_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.json"
            info = export_human_review(REVIEW, output)
            self.assertEqual(info["pairs"], 12)
            self.assertEqual(info["pending"], 12)
            data = json.loads(output.read_text(encoding="utf-8"))
            for entry in data:
                self.assertNotIn("condition", entry)
                self.assertNotIn("current", entry)
                self.assertNotIn("purpose", [key.lower() for key in entry.keys()])
                self.assertFalse(entry["human_review"]["reviewed"])


if __name__ == "__main__":
    unittest.main()
