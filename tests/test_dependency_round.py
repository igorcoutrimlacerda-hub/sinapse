"""Testes da rodada de dependências condicionais (modo simulado)."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.dependency_round import run_dependency_round


class DependencyRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_dependency_round(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_calls_and_verdicts(self) -> None:
        self.assertEqual(self.result["cases"], 6)
        self.assertEqual(self.result["calls"], 10)
        self.assertEqual(self.result["comparisons"]["conservative_preserved"], 2)
        self.assertEqual(self.result["comparisons"]["conservative_harmed"], 0)
        self.assertEqual(self.result["comparisons"]["no_effect"], 4)

    def test_bike_named_omits_dependency_only(self) -> None:
        entry = self.entries["dep_bike_named"]
        self.assertEqual(entry["reference_ids"], ["p1", "p2"])
        self.assertEqual(entry["conservative_ids"], ["p2"])
        self.assertTrue(entry["local_only_dependency_change"])
        self.assertEqual(entry["verdict"], "preservou")

    def test_ambiguous_and_two_decisions_keep_link(self) -> None:
        for case in ("dep_two_entities", "dep_two_decisions"):
            entry = self.entries[case]
            self.assertFalse(entry["conservative_omitted"])
            self.assertEqual(entry["conservative_ids"], entry["reference_ids"])

    def test_two_purposes_preserve_both_parts(self) -> None:
        entry = self.entries["dep_two_purposes"]
        self.assertTrue(entry["conservative_omitted"])
        checks = entry["conditions"]["conservative"]["checks"]["purposes"]
        self.assertTrue(checks["m"]["ok"])
        self.assertTrue(checks["p"]["ok"])

    def test_redundant_case_has_no_effect(self) -> None:
        entry = self.entries["dep_redundant"]
        self.assertFalse(entry["conservative_omitted"])
        self.assertEqual(entry["verdict"], "sem_efeito_na_requisicao")


if __name__ == "__main__":
    unittest.main()
