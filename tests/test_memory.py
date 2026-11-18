"""Testes da memória: pontos, grupos, semente, vínculo, evidência e atenção."""

from __future__ import annotations

import unittest
from pathlib import Path

from sinapse.budget import is_duplicate
from sinapse.grouping import build_groups
from sinapse.memory import link_between, recall, requires_seed
from sinapse.models import Message, load_messages
from sinapse.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo_conversation.json"
CONFIG = ROOT / "data" / "config.json"

POR_QUE = "Por que eu escolhi o Serum 2?"
COMO = "Como eu testei o bass?"
QUANDO = "Quando escolhi o Serum 2?"
PIANO = "Por que eu escolhi o Piano V?"
SERUM_CONTEXT = ["Estávamos falando do Serum 2 e do bass."]
PIANO_CONTEXT = ["Estávamos falando do Piano V e do jazz."]


class DemoMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages = load_messages(DATA)
        cls.config = load_config(CONFIG)
        cls.by_id = {m.id: m for m in cls.messages}

    def run_recall(self, query: str, context: list[str], **kwargs):
        return recall(self.messages, query, context, self.config, **kwargs)

    def test_each_message_is_one_point_with_original_identity(self) -> None:
        self.assertEqual(len(self.messages), 6)
        self.assertEqual(len({m.id for m in self.messages}), 6)
        self.assertEqual(
            self.by_id["m1"].text, "Escolhi o Serum 2 para fazer o bass."
        )
        self.assertEqual(self.by_id["m2"].reply_to, "m1")
        self.assertEqual(self.by_id["m2"].position, 2)
        self.assertIsNotNone(self.by_id["m1"].sent_at)
        self.assertEqual(self.by_id["m1"].values("por_que"), ())
        self.assertEqual(self.by_id["m2"].values("o_que"), ())

    def test_overlapping_groups_across_dimensions(self) -> None:
        index = build_groups(self.messages, threshold=0.33)
        m1_groups = index.groups_for("m1")
        m3_groups = index.groups_for("m3")
        self.assertGreaterEqual(len(m1_groups), 3)
        self.assertGreaterEqual(len(m3_groups), 4)
        self.assertEqual(
            {g.dimension for g in m1_groups},
            {"o_que", "para_que", "quem"},
        )
        o_que_groups = [g for g in index.groups if g.dimension == "o_que"]
        together = [g for g in o_que_groups if {"m1", "m3"} <= set(g.message_ids)]
        self.assertEqual(len(together), 1)
        por_que_groups = [g for g in index.groups if g.dimension == "por_que"]
        self.assertFalse(any({"m2", "m6"} <= set(g.message_ids) for g in por_que_groups))

    def test_group_threshold_is_adjustable(self) -> None:
        strict = build_groups(self.messages, threshold=0.50)
        relaxed = build_groups(self.messages, threshold=0.33)
        strict_oque = [g for g in strict.groups if g.dimension == "o_que"]
        relaxed_oque = [g for g in relaxed.groups if g.dimension == "o_que"]
        self.assertFalse(any({"m1", "m3"} <= set(g.message_ids) for g in strict_oque))
        self.assertTrue(any({"m1", "m3"} <= set(g.message_ids) for g in relaxed_oque))

    def test_por_que_prioritizes_linked_motive(self) -> None:
        result = self.run_recall(POR_QUE, SERUM_CONTEXT)
        self.assertEqual(result.seed.id, "m1")
        self.assertEqual(result.focus, "por_que")
        self.assertEqual([c.message.id for c in result.evidence], ["m2"])
        m2 = next(c for c in result.candidates if c.message.id == "m2")
        self.assertAlmostEqual(m2.distance, 0.117647, places=5)
        self.assertAlmostEqual(m2.attention, 0.90, places=6)
        self.assertEqual(m2.association_source, "link")
        self.assertAlmostEqual(m2.association_content, 0.0)
        self.assertEqual(m2.evidence["por_que"], 1.0)
        self.assertIsNone(m2.evidence["o_que"])
        self.assertIsNone(m2.evidence["para_que"])
        m6 = next(c for c in result.candidates if c.message.id == "m6")
        self.assertEqual(m6.evidence["por_que"], 0.0)
        self.assertFalse(m6.within_radius)

    def test_como_prioritizes_linked_procedure_and_keeps_zero_attention_candidate(self) -> None:
        result = self.run_recall(COMO, SERUM_CONTEXT)
        self.assertEqual(result.seed.id, "m1")
        self.assertEqual(result.focus, "como")
        self.assertEqual([c.message.id for c in result.evidence], ["m3"])
        m2 = next(c for c in result.candidates if c.message.id == "m2")
        self.assertTrue(m2.within_radius)
        self.assertIsNone(m2.evidence["como"])
        self.assertEqual(m2.attention, 0.0)
        m3 = next(c for c in result.candidates if c.message.id == "m3")
        self.assertAlmostEqual(m3.distance, 0.274510, places=5)
        self.assertAlmostEqual(m3.attention, 0.90, places=6)
        self.assertEqual(m3.evidence["como"], 1.0)
        self.assertAlmostEqual(m3.evidence["o_que"], 1 / 3, places=6)
        self.assertAlmostEqual(m3.evidence["para_que"], 1 / 3, places=6)

    def test_quando_has_no_temporal_evidence_despite_sent_at(self) -> None:
        result = self.run_recall(QUANDO, SERUM_CONTEXT)
        self.assertEqual(result.seed.id, "m1")
        self.assertEqual(result.focus, "quando")
        self.assertEqual(result.evidence, ())
        m2 = next(c for c in result.candidates if c.message.id == "m2")
        self.assertTrue(m2.within_radius)
        self.assertIsNone(m2.evidence["quando"])
        self.assertEqual(m2.attention, 0.0)
        self.assertIsNotNone(self.by_id["m2"].sent_at)
        self.assertEqual(self.by_id["m1"].values("quando"), ())

    def test_other_topic_piano_uses_same_general_rules(self) -> None:
        result = self.run_recall(PIANO, PIANO_CONTEXT)
        self.assertEqual(result.seed.id, "m5")
        self.assertEqual(result.focus, "por_que")
        self.assertEqual([c.message.id for c in result.evidence], ["m6"])
        m6 = next(c for c in result.candidates if c.message.id == "m6")
        self.assertAlmostEqual(m6.distance, 0.117647, places=5)
        self.assertEqual(m6.association_source, "link")
        self.assertEqual(m6.evidence["por_que"], 1.0)

    def test_reply_link_is_required_for_short_motive(self) -> None:
        linked = self.run_recall(POR_QUE, SERUM_CONTEXT)
        unlinked = self.run_recall(POR_QUE, SERUM_CONTEXT, use_links=False)
        self.assertIn("m2", [c.message.id for c in linked.evidence])
        self.assertNotIn("m2", [c.message.id for c in unlinked.evidence])
        m2 = next(c for c in unlinked.candidates if c.message.id == "m2")
        self.assertFalse(m2.within_radius)
        self.assertGreater(m2.distance, 0.70)
        self.assertEqual(m2.association_source, "none")
        self.assertEqual(m2.evidence["por_que"], 0.0)

    def test_m4_is_outside_radius_in_all_queries(self) -> None:
        for query, context in (
            (POR_QUE, SERUM_CONTEXT),
            (COMO, SERUM_CONTEXT),
            (QUANDO, SERUM_CONTEXT),
            (PIANO, PIANO_CONTEXT),
        ):
            result = self.run_recall(query, context)
            m4 = next(c for c in result.candidates if c.message.id == "m4")
            self.assertFalse(m4.within_radius, query)

    def test_short_reply_depends_on_seed(self) -> None:
        self.assertTrue(requires_seed(self.by_id["m2"], self.by_id["m1"]))
        self.assertFalse(requires_seed(self.by_id["m3"], self.by_id["m1"]))
        self.assertTrue(requires_seed(self.by_id["m6"], self.by_id["m5"]))
        self.assertFalse(requires_seed(self.by_id["m4"], self.by_id["m1"]))
        self.assertTrue(link_between(self.by_id["m2"], self.by_id["m1"]))

    def test_negation_and_numbers_are_not_deduplicated(self) -> None:
        coords = {"o_que": ("Serum 2",)}
        positive = Message("a", "Escolhi o Serum 2", "user", "c", 1, None, None, coords)
        negative = Message("b", "Não escolhi o Serum 2", "user", "c", 2, None, None, coords)
        another_number = Message("c", "Escolhi o Serum 3", "user", "c", 3, None, None, {"o_que": ("Serum 3",)})
        paraphrase = Message("d", "Optei pelo Serum 2", "user", "c", 4, None, None, coords)
        exact = Message("e", "Escolhi o Serum 2.", "user", "c", 5, None, None, coords)
        self.assertFalse(is_duplicate(positive, negative))
        self.assertFalse(is_duplicate(positive, another_number))
        self.assertFalse(is_duplicate(positive, paraphrase))
        self.assertTrue(is_duplicate(positive, exact))

    def test_reference_resolution_and_seed_for_all_queries(self) -> None:
        reference = self.run_recall("E por que eu escolhi ele?", SERUM_CONTEXT)
        self.assertEqual(reference.seed.id, "m1")
        self.assertEqual(reference.references.pronouns, ("ele",))
        self.assertEqual(reference.references.source_turn_index, 0)
        self.assertIn("serum", reference.references.referent_tokens)
        for query, context in (
            (POR_QUE, SERUM_CONTEXT),
            (COMO, SERUM_CONTEXT),
            (QUANDO, SERUM_CONTEXT),
        ):
            self.assertEqual(self.run_recall(query, context).seed.id, "m1")

    def test_recall_is_deterministic(self) -> None:
        first = self.run_recall(COMO, SERUM_CONTEXT)
        second = self.run_recall(COMO, SERUM_CONTEXT)
        self.assertEqual(
            [(c.message.id, c.distance, c.attention) for c in first.candidates],
            [(c.message.id, c.distance, c.attention) for c in second.candidates],
        )


if __name__ == "__main__":
    unittest.main()
