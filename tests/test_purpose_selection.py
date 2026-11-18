"""Testes da seleção por tema e propósito."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.purpose_selection import run_purpose_benchmark


class PurposeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_purpose_benchmark(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_purpose_coverage_beats_reference(self) -> None:
        self.assertEqual(self.result["purpose"]["purpose_coverage"], 1.0)
        self.assertEqual(self.result["purpose"]["all_purposes_cases"], 7)
        self.assertLess(
            self.result["reference"]["purpose_coverage"],
            self.result["purpose"]["purpose_coverage"],
        )

    def test_dependencies_and_distractions(self) -> None:
        self.assertEqual(self.result["purpose"]["deps_lost"], 0)
        self.assertEqual(self.result["purpose"]["distractions"], 0)
        self.assertGreaterEqual(
            self.result["reference"]["deps_lost"],
            self.result["purpose"]["deps_lost"],
        )

    def test_combined_pack_covers_two_purposes(self) -> None:
        entry = self.entries["ev_pack_both"]
        self.assertFalse(entry["reference"]["all_purposes"])
        self.assertTrue(entry["purpose"]["all_purposes"])
        self.assertIn("m7", entry["purpose"]["selected"])
        self.assertIn("m8", entry["purpose"]["selected"])

    def test_distinct_themes_cover_two_purposes(self) -> None:
        entry = self.entries["ev_two_themes"]
        self.assertFalse(entry["reference"]["all_purposes"])
        self.assertTrue(entry["purpose"]["all_purposes"])
        self.assertIn("m6", entry["purpose"]["selected"])
        self.assertIn("m8", entry["purpose"]["selected"])

    def test_contrast_omits_pack_quantity_for_serum_motive(self) -> None:
        entry = self.entries["ev_serum_contrast"]
        self.assertNotIn("m8", entry["purpose"]["selected"])
        self.assertEqual(entry["purpose"]["distractions"], [])

    def test_history_is_not_repeated(self) -> None:
        entry = self.entries["ev_history_bike"]
        self.assertEqual(entry["purpose"]["selected"], [])
        self.assertEqual(entry["purpose"]["memory_tokens"], 0)
        self.assertTrue(entry["purpose"]["all_purposes"])


if __name__ == "__main__":
    unittest.main()
