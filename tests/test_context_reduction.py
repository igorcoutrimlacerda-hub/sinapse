"""Testes da redução de contexto: preservação e economia."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.context_reduction import run_context_reduction


class ContextReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_context_reduction(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_expected_evidence_is_preserved(self) -> None:
        self.assertEqual(
            self.result["current"]["expected"], self.result["reduced"]["expected"]
        )
        self.assertEqual(self.result["reduced"]["expected"], 10)

    def test_functionless_and_tokens_are_reduced(self) -> None:
        self.assertLess(
            self.result["reduced"]["functionless"],
            self.result["current"]["functionless"],
        )
        self.assertLess(
            self.result["reduced"]["memory_tokens"],
            self.result["current"]["memory_tokens"],
        )
        self.assertLessEqual(
            self.result["reduced"]["tokens"], self.result["current"]["tokens"]
        )

    def test_short_reply_keeps_seed_and_context(self) -> None:
        for case in ("ev_follow", "ev_follow_piano", "ev_price_link"):
            entry = self.entries[case]
            self.assertIn(entry["reduced"]["selected"][0], {"p1", "m5", "m1"})
            self.assertGreaterEqual(len(entry["reduced"]["selected"]), 2)
            self.assertEqual(entry["reduced"]["context"], entry["current"]["context"])

    def test_redundant_seed_and_functionless_items_are_removed(self) -> None:
        correction = self.entries["ev_correction"]
        self.assertIn("m3", correction["reduced"]["removed"])
        self.assertIn("m7", correction["reduced"]["removed"])
        many = self.entries["ev_many_links"]
        self.assertIn("m30", many["reduced"]["removed"])
        history = self.entries["ev_history"]
        self.assertEqual(history["reduced"]["selected"], [])
        self.assertEqual(history["reduced"]["memory_tokens"], 0)

    def test_no_necessary_context_or_evidence_loss(self) -> None:
        self.assertEqual(self.result["tradeoffs"], [])
        for entry in self.result["entries"]:
            self.assertFalse(entry["reduced"]["necessary_context_loss"])

    def test_self_sufficient_evidence_keeps_its_seed(self) -> None:
        entry = self.entries["ev_self_sufficient"]
        self.assertEqual(entry["reduced"]["selected"], ["m20"])
        self.assertEqual(entry["reduced"]["expected"], ["m20"])


if __name__ == "__main__":
    unittest.main()
