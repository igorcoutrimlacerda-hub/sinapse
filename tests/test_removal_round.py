"""Testes da rodada de remoções dirigidas (modo simulado)."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.removal_round import run_removal_round


class RemovalRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_removal_round(load_config("data/config.json"))

    def test_call_budget_and_reuse(self) -> None:
        self.assertEqual(self.result["planned_calls"], 20)
        self.assertEqual(self.result["executed_calls"], 12)
        self.assertEqual(self.result["cases"], 5)

    def test_confirmed_removals_and_no_effect(self) -> None:
        entries = {entry["case"]: entry for entry in self.result["entries"]}
        bike = {item["item"]: item for item in entries["plan_bike_two"]["comparisons"]}
        self.assertEqual(bike["p2"]["verdict"], "confirmada")
        self.assertFalse(bike["p2"]["result"]["checks"]["purposes"]["m"]["ok"])
        self.assertTrue(bike["p2"]["result"]["checks"]["purposes"]["p"]["ok"])
        priority = {item["item"]: item for item in entries["plan_priority"]["comparisons"]}
        self.assertEqual(priority["m7"]["verdict"], "sem_efeito_na_requisicao")
        self.assertTrue(priority["m7"]["same_request"])

    def test_dependency_removal_is_evaluated(self) -> None:
        entry = {entry["case"]: entry for entry in self.result["entries"]}["plan_bike_two"]
        item = {item["item"]: item for item in entry["comparisons"]}["p1"]
        self.assertTrue(item["present_in_purpose"])
        self.assertIn(item["verdict"], {"confirmada", "falhou"})

    def test_removal_does_not_reinsert(self) -> None:
        entry = {entry["case"]: entry for entry in self.result["entries"]}["plan_two_themes"]
        item = {item["item"]: item for item in entry["comparisons"]}["m6"]
        self.assertNotIn("m6", item["result"]["selected"])
        self.assertIn("b3", item["result"]["selected"])

    def test_history_and_restriction_have_no_request_effect(self) -> None:
        for case in ("plan_history", "plan_restricted"):
            entry = {entry["case"]: entry for entry in self.result["entries"]}[case]
            for comparison in entry["comparisons"]:
                self.assertEqual(comparison["verdict"], "sem_efeito_na_requisicao")
                self.assertTrue(comparison["same_request"])


if __name__ == "__main__":
    unittest.main()
