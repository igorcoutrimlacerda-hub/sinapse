"""Validação local dos pares preparados para a comparação real."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.deepseek import load_deepseek_settings
from sinapse.models import load_messages
from sinapse.resumption import cases
from sinapse.resumption_validation import validate_real_comparison


class ResumptionValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.settings = load_deepseek_settings(cls.config)
        cls.result = validate_real_comparison(
            config=cls.config, settings=cls.settings
        )
        cls.pairs = {pair["case"]: pair for pair in cls.result["pairs"]}

    def test_all_pairs_are_valid_and_frozen(self) -> None:
        summary = self.result["summary"]
        self.assertEqual(summary["pairs"], 10)
        self.assertEqual(summary["pair_valid"], 10)
        self.assertEqual(summary["selection_matches_frozen"], 10)
        self.assertEqual(summary["single_integrity"], 10)
        self.assertEqual(summary["multipurpose_integrity"], 10)
        self.assertFalse(summary["real_evaluation_available"])

    def test_multipurpose_completes_all_elements(self) -> None:
        summary = self.result["summary"]
        self.assertEqual(summary["multipurpose_complete"], 10)
        self.assertEqual(summary["single_complete"], 7)

    def test_references_are_sent_and_support_their_elements(self) -> None:
        for pair in self.result["pairs"]:
            for condition in ("single", "multipurpose"):
                item = pair[condition]
                self.assertTrue(item["references_sent"], pair["case"])
                self.assertEqual(
                    item["reference_failures"], [], pair["case"]
                )
                names = list(item["checks"]["components"]) + (
                    ["condition"] if "condition" in item["reference_support"] else []
                )
                for name in names:
                    if name in item["missing_elements"]:
                        continue
                    self.assertTrue(
                        item["reference_support"][name],
                        f"{pair['case']}:{name}",
                    )

    def test_dunas_condition_and_both_paths(self) -> None:
        pair = self.pairs["proximo_condicionado"]
        for condition in ("single", "multipurpose"):
            item = pair[condition]
            self.assertTrue(item["conditions_ok"], condition)
            self.assertEqual(
                item["condition_paths"],
                {
                    "condition": True,
                    "export_path": True,
                    "fallback_path": True,
                },
            )

    def test_north_and_south_keep_project_attribution(self) -> None:
        corpus = {
            message.id: message
            for message in load_messages("benchmark/resumption_sessions.json")
        }
        pairs = {
            case.id: case
            for case in cases()
            if case.id.startswith("intercalado_")
        }
        for case_id, case in pairs.items():
            pair = self.pairs[case_id]
            for condition in ("single", "multipurpose"):
                for message_id in pair[condition]["selected"]:
                    self.assertEqual(
                        corpus[message_id].conversation_id,
                        case.conversation_id,
                        f"{case_id}:{message_id}",
                    )
                self.assertTrue(pair[condition]["project_ok"], case_id)
                self.assertEqual(
                    pair[condition]["wrong_project_selected"], [], case_id
                )

    def test_existing_review_files_are_preserved(self) -> None:
        expected = {
            "benchmark_out_resumption/resumption_review.json": 6,
            "benchmark_out_resumption_round2/resumption_review.json": 18,
            "benchmark_out_resumption_round3/resumption_review.json": 30,
        }
        for path, count in expected.items():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(data), count, path)
            self.assertTrue(
                all(not item["human_review"]["reviewed"] for item in data)
            )


if __name__ == "__main__":
    unittest.main()
