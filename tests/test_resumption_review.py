"""Revisão cega: preparação, bloqueio antes das notas e consolidação."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.resumption_review import consolidate_review, prepare_blind_review

REVIEW = "benchmark_out_resumption_real/real_review_blind.json"
RESULTS = "benchmark_out_resumption_real/real_results.json"


class HumanReviewTest(unittest.TestCase):
    def test_blind_review_layout_and_hidden_conditions(self) -> None:
        review = json.loads(Path(REVIEW).read_text(encoding="utf-8"))
        self.assertEqual(len(review), 10)
        for pair in review:
            self.assertNotIn("condition", pair)
            self.assertNotIn("conditions", pair)
            self.assertTrue(pair["sources"])
            self.assertTrue(pair["applicable_elements"])

    def test_prepare_creates_blank_human_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared.json"
            prepare_blind_review(REVIEW, RESULTS, target)
            prepared = json.loads(target.read_text(encoding="utf-8"))
        for pair in prepared:
            self.assertFalse(pair["human_review"]["reviewed"])
            for label in ("A", "B"):
                for name in pair["applicable_elements"]:
                    field = pair["human"][label][name]
                    self.assertIsNone(field["score"])
                    self.assertEqual(field["notes"], "")
                    if name == "decision":
                        self.assertIsNone(field["current_identified"])
                        self.assertIsNone(field["reference_supports"])
                        self.assertIsNone(field["old_or_ambiguous"])

    def test_consolidation_refuses_blank_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared.json"
            prepare_blind_review(REVIEW, RESULTS, target)
            result = consolidate_review(target, RESULTS)
        self.assertFalse(result["complete"])
        self.assertFalse(result["revealed"])
        self.assertTrue(result["missing"])
        self.assertNotIn("revealed_conditions", result)

    def test_current_review_consolidates_with_preferences(self) -> None:
        result = consolidate_review(REVIEW, RESULTS)
        self.assertTrue(result["complete"])
        self.assertTrue(result["revealed"])
        self.assertEqual(result["reviewed_pairs"], 10)
        self.assertEqual(len(result["revealed_conditions"]), 10)
        self.assertEqual(
            set(result["human_summary"]),
            {"sinapse", "sinapse_multipurpose"},
        )
        self.assertEqual(
            result["preferences"]["sinapse_multipurpose"]["wins"], 6
        )
        self.assertEqual(result["preferences"]["sinapse"]["wins"], 3)
        self.assertIn("assistida por IA", result["review_method"])
        self.assertTrue(result["divergences"])

    def test_consolidation_reveals_after_all_notes(self) -> None:
        review = json.loads(Path(REVIEW).read_text(encoding="utf-8"))
        for pair in review:
            for label in ("A", "B"):
                for name in pair["applicable_elements"]:
                    field = pair["human"][label][name]
                    field["score"] = 1
                    if name == "decision":
                        field["current_identified"] = True
                        field["reference_supports"] = True
                        field["old_or_ambiguous"] = False
            pair["human_review"]["reviewed"] = True
            pair["human_review"]["preferred"] = "empate"
        with tempfile.TemporaryDirectory() as directory:
            review_path = Path(directory) / "review.json"
            out_path = Path(directory) / "consolidated.json"
            review_path.write_text(
                json.dumps(review, ensure_ascii=False), encoding="utf-8"
            )
            result = consolidate_review(
                review_path, RESULTS, out_path
            )
            self.assertTrue(result["complete"])
            self.assertTrue(result["revealed"])
            self.assertEqual(result["reviewed_pairs"], 10)
            self.assertEqual(len(result["revealed_conditions"]), 10)
            self.assertEqual(set(result["human_summary"]), {"sinapse", "sinapse_multipurpose"})
            self.assertEqual(result["human_summary"]["sinapse"]["complete"], 10)
            self.assertTrue(Path(out_path).exists())
            self.assertTrue(result["divergences"])

    def test_prepare_is_idempotent_on_existing_answers(self) -> None:
        review = json.loads(Path(REVIEW).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prepared.json"
            prepare_blind_review(REVIEW, RESULTS, target)
            prepared = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(
                [pair["answers"] for pair in prepared],
                [pair["answers"] for pair in review],
            )
            self.assertTrue(
                all(
                    pair["human_review"]["reviewed"] is False
                    for pair in prepared
                )
            )


if __name__ == "__main__":
    unittest.main()
