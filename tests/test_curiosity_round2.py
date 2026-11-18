"""Testes da rodada 2: sinal de vínculo, regressão e métricas separadas."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.curiosity_round2 import run_round2


class CuriosityRound2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_round2(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_ev_follow_is_recovered_with_link_signal(self) -> None:
        entry = self.entries["ev_follow"]
        self.assertEqual(entry["old"]["action"], "stop")
        self.assertEqual(entry["link"]["action"], "follow_links")
        self.assertIn("p2", entry["link"]["expected"])
        self.assertEqual(entry["link"]["context"], ["p1"])

    def test_ev_correction_is_preserved(self) -> None:
        entry = self.entries["ev_correction"]
        self.assertEqual(entry["link"]["action"], "correction")
        self.assertIn("m10", entry["link"]["expected"])

    def test_link_signal_has_measurable_cost(self) -> None:
        self.assertGreaterEqual(self.result["signal_avg_cost_ms"], 0.0)
        features = self.entries["ev_follow"]["features_link"]
        self.assertGreater(features["respostas_diretas"], 0.0)
        self.assertIn("respostas_diretas", features)

    def test_irrelevant_categories_are_separate(self) -> None:
        for condition in ("off", "fixed", "old", "more", "link"):
            item = self.result["summary"][condition]
            self.assertIn("expected", item)
            self.assertIn("context", item)
            self.assertIn("functionless", item)
            self.assertGreaterEqual(item["tokens"], item["memory_tokens"])

    def test_no_gain_is_reported_over_extra_searches(self) -> None:
        for condition in ("fixed", "old", "more", "link"):
            item = self.result["summary"][condition]
            self.assertLessEqual(item["no_gain_searches"], item["extra_searches"])
            self.assertLessEqual(item["queries_triggered"], item["cases"])

    def test_link_signal_does_not_break_sensitivity(self) -> None:
        entry = self.entries["ev_restricted"]
        self.assertEqual(entry["link"]["action"], "stop")
        self.assertEqual(entry["link"]["expected"], [])


if __name__ == "__main__":
    unittest.main()
