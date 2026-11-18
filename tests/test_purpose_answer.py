"""Testes da avaliação de resposta por propósito (modo simulado)."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.purpose_answer import run_purpose_answer


class PurposeAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_purpose_answer(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_cases_calls_and_conditions(self) -> None:
        self.assertEqual(self.result["cases"], 12)
        self.assertEqual(self.result["calls"], 26)
        self.assertEqual(self.result["conditions"]["purpose"]["cases"], 12)
        self.assertEqual(self.result["conditions"]["removal"]["cases"], 2)

    def test_purpose_beats_reference_on_coverage(self) -> None:
        reference = self.result["conditions"]["reference"]
        purpose = self.result["conditions"]["purpose"]
        self.assertGreater(purpose["purposes_ok"], reference["purposes_ok"])
        self.assertEqual(purpose["dependencies_lost"], 0)
        self.assertEqual(purpose["unsupported"], 0)
        self.assertEqual(purpose["sensitive"], 0)
        self.assertEqual(purpose["all_ok"], 12)

    def test_removal_loses_target_but_preserves_other(self) -> None:
        bike = self.entries["new_bike_two"]["conditions"]["removal"]
        self.assertFalse(bike["checks"]["purposes"]["m"]["ok"])
        self.assertTrue(bike["checks"]["purposes"]["p"]["ok"])
        themes = self.entries["new_two_themes"]["conditions"]["removal"]
        self.assertFalse(themes["checks"]["purposes"]["m"]["ok"])
        self.assertTrue(themes["checks"]["purposes"]["p"]["ok"])

    def test_priority_case_keeps_motive(self) -> None:
        entry = self.entries["new_priority"]
        self.assertTrue(entry["conditions"]["purpose"]["checks"]["priority_ok"])

    def test_history_and_restricted_use_no_memory(self) -> None:
        history = self.entries["new_history_price"]["conditions"]["purpose"]
        restricted = self.entries["new_restricted_price"]["conditions"]["purpose"]
        self.assertEqual(history["memory_tokens"], 0)
        self.assertEqual(restricted["memory_tokens"], 0)
        self.assertFalse(restricted["checks"]["unsupported"])
        self.assertFalse(
            any(value.get("sensitive", False) for value in restricted["checks"]["purposes"].values())
        )


if __name__ == "__main__":
    unittest.main()
