"""Testes do rastreamento de perdas em escala e da busca com histórico recente."""

from __future__ import annotations

import unittest
from collections import defaultdict

from sinapse.config import load_config
from sinapse.models import Message
from sinapse.scale_benchmark import ScaleQuery, _tokens, build_shortlist
from sinapse.scale_recovery import NEW_QUERIES, run_recovery


class ScaleRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_recovery(
            scale=400, seed=42, candidate_k=200, config=load_config()
        )

    def test_reproduces_original_settings(self) -> None:
        self.assertEqual(self.result["scale"], 400)
        self.assertEqual(self.result["seed"], 42)
        self.assertEqual(self.result["candidate_k"], 200)
        self.assertEqual(self.result["budget"], 160)
        self.assertEqual(self.result["messages"], 400)

    def test_rows_cover_regressions_and_new_families(self) -> None:
        rows = self.result["rows"]
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum(row["group"] == "regression" for row in rows), 8)
        self.assertEqual(sum(row["group"] == "new" for row in rows), 3)
        self.assertEqual(
            [row["query"] for row in rows if row["group"] == "new"],
            [query.id for query in NEW_QUERIES],
        )

    def test_aggregates_only_count_queries_with_evidence(self) -> None:
        regression = self.result["aggregates"]["regression"]
        new = self.result["aggregates"]["new"]
        self.assertEqual(regression["queries"], 8)
        self.assertEqual(new["queries"], 3)
        for item in (regression, new):
            self.assertLessEqual(item["final_before"], item["queries"])
            self.assertLessEqual(item["final_after"], item["queries"])
            self.assertLessEqual(item["improved"] + item["regressed"], item["queries"])

    def test_traces_only_exist_for_missing_queries(self) -> None:
        missing = {
            row["query"]
            for row in self.result["rows"]
            if row["group"] == "regression" and not row["final_before"]
        }
        traced = {trace["query"] for trace in self.result["traces"]}
        self.assertEqual(traced, missing)
        for trace in self.result["traces"]:
            self.assertFalse(trace["in_final"])
            self.assertIn(
                trace["first_loss_stage"],
                {"corte_candidatos", "quadro", "orcamento"},
            )


class ExtraContextTest(unittest.TestCase):
    def test_extra_context_adds_search_tokens(self) -> None:
        messages = [
            Message(id="a", text="somente alfa", author="user", conversation_id="c1", position=1),
            Message(id="b", text="beta raro", author="user", conversation_id="c1", position=2),
        ]
        postings: dict[str, list[int]] = defaultdict(list)
        for index, message in enumerate(messages):
            for token in _tokens(message):
                postings[token].append(index)
        id_to_index = {message.id: index for index, message in enumerate(messages)}
        query = ScaleQuery("q", "alfa", (), ("a",), ())

        without = build_shortlist(
            messages, query, 2, postings, {}, id_to_index
        )
        with_context = build_shortlist(
            messages, query, 2, postings, {}, id_to_index, extra_context=["beta"]
        )

        self.assertEqual(without.ids, frozenset({"a"}))
        self.assertEqual(with_context.ids, frozenset({"a", "b"}))


if __name__ == "__main__":
    unittest.main()
