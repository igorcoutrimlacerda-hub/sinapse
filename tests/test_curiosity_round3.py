"""Testes da rodada 3: vínculo compatível, contradição do piano e métricas."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.curiosity_round3 import run_round3


class CuriosityRound3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_round3(load_config("data/config.json"))
        cls.entries = {entry["case"]: entry for entry in cls.result["entries"]}

    def test_follow_piano_follows_link_to_m6(self) -> None:
        entry = self.entries["ev_follow_piano"]
        self.assertEqual(entry["more"]["action"], "follow_links")
        self.assertEqual(entry["compatible"]["action"], "follow_links")
        self.assertIn("m6", entry["compatible"]["expected"])
        self.assertNotIn("m6", entry["all_actions"]["other_theme"]["expected"])
        self.assertEqual(entry["fixed"]["action"], "other_theme")

    def test_link_compatible_signal_is_pre_search_and_sensitive(self) -> None:
        features = self.entries["ev_follow"]["features_compatible"]
        self.assertGreater(features["vinculo_compativel"], 0.0)
        self.assertGreater(self.result["index_avg_query_ms"], 0.0)
        self.assertGreaterEqual(self.result["index_avg_entries"], 0.0)
        restricted = self.entries["ev_restricted"]
        self.assertEqual(restricted["compatible"]["action"], "stop")
        self.assertEqual(restricted["compatible"]["expected"], [])

    def test_more_training_remains_reference(self) -> None:
        more = self.result["summary"]["more"]
        compatible = self.result["summary"]["compatible"]
        self.assertGreaterEqual(more["expected"], compatible["expected"])
        self.assertGreaterEqual(more["expected"], self.result["summary"]["fixed"]["expected"])
        self.assertEqual(more["no_gain_searches"], compatible["no_gain_searches"])

    def test_irrelevant_categories_and_token_costs(self) -> None:
        for condition in ("off", "fixed", "more", "compatible"):
            item = self.result["summary"][condition]
            self.assertIn("expected", item)
            self.assertIn("context", item)
            self.assertIn("functionless", item)
            self.assertGreaterEqual(item["tokens"], item["memory_tokens"])

    def test_regressions_bicycle_and_correction(self) -> None:
        follow = self.entries["ev_follow"]
        correction = self.entries["ev_correction"]
        self.assertEqual(follow["compatible"]["action"], "follow_links")
        self.assertIn("p2", follow["compatible"]["expected"])
        self.assertEqual(correction["compatible"]["action"], "correction")
        self.assertIn("m10", correction["compatible"]["expected"])


if __name__ == "__main__":
    unittest.main()
