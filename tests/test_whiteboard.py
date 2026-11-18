"""Testes do agente com quadro branco e atenção temática."""

from __future__ import annotations

import unittest
from pathlib import Path

from sinapse.budget import RecentTurn
from sinapse.chat import EstimatedChatCounter, build_base_messages
from sinapse.config import load_config
from sinapse.deepseek import load_deepseek_settings
from sinapse.models import load_messages
from sinapse.whiteboard import MemoryAgent

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_conversation.json"
CONFIG = ROOT / "data" / "config.json"

TURNS = (
    ("c1", "Por que eu escolhi o preset Serum 2?", ("Falamos do Serum 2 e do bass.",)),
    ("c1", "E como eu testei?", ("Por que eu escolhi o preset Serum 2?",)),
    ("c1", "Passei a tarde jogando xadrez.", ("E como eu testei?",)),
    (
        "c1",
        "Voltando ao preset: por que eu escolhi ele?",
        ("Passei a tarde jogando xadrez.",),
    ),
    (
        "c1",
        "Como eu divulguei o pack do bass?",
        ("Voltando ao preset: por que eu escolhi ele?",),
    ),
    ("c1", "E aquilo que eu preparei?", ("Como eu divulguei o pack do bass?",)),
)


class WhiteboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages = load_messages(DATA)
        cls.config = load_config(CONFIG)
        cls.settings = load_deepseek_settings(cls.config)

    def make_agent(self) -> MemoryAgent:
        return MemoryAgent(self.messages, self.config)

    def run_sequence(self, agent: MemoryAgent):
        return [agent.before_response(cid, query, recent) for cid, query, recent in TURNS]

    def test_sequence_enters_keeps_and_leaves(self) -> None:
        agent = self.make_agent()
        turns = self.run_sequence(agent)

        first, second, third, fourth, fifth, sixth = turns
        self.assertEqual(set(first.delta.entered), {"m1", "m2"})
        self.assertEqual(first.budget.diagnostics.selected_ids, ("m2",))

        self.assertEqual(set(second.delta.entered), {"m3"})
        self.assertEqual(set(second.delta.stayed), {"m1"})
        self.assertEqual(set(second.delta.left), {"m2"})
        self.assertEqual(second.delta.left_reasons["m2"], "no_focus_evidence")
        self.assertEqual(second.budget.diagnostics.selected_ids, ("m3",))
        self.assertEqual(
            set(item.message_id for item in second.board), {"m1", "m3"}
        )

        self.assertEqual(set(third.delta.left), {"m1", "m3"})
        self.assertEqual(set(third.delta.entered), {"m4"})
        self.assertEqual(
            set(item.message_id for item in third.board), {"m4"}
        )
        self.assertEqual(third.budget.diagnostics.selected_ids, ())

        self.assertEqual(set(fourth.delta.entered), {"m1", "m2"})
        self.assertEqual(set(fourth.delta.left), {"m4"})
        self.assertIn("m2", fourth.budget.diagnostics.selected_ids)
        second_item = next(item for item in fourth.board if item.message_id == "m2")
        self.assertAlmostEqual(second_item.evidence_focus or 0.0, 1.0)

        self.assertAlmostEqual(fifth.themes.weights()["pack"], 1.0)
        self.assertAlmostEqual(fifth.themes.weights()["bass"], 1.0)
        self.assertEqual(set(fifth.delta.entered), {"m7", "m9"})
        self.assertEqual(set(fifth.delta.left), {"m1", "m2"})
        self.assertIn("m7", fifth.budget.diagnostics.selected_ids)
        self.assertGreater(fifth.budget.diagnostics.memory_tokens, 0)

        self.assertTrue(sixth.themes.ambiguous)
        evidence_items = [item for item in sixth.board if item.role == "evidência"]
        self.assertEqual(len(evidence_items), 1)
        self.assertLessEqual(len(sixth.budget.diagnostics.selected_ids), 1)
        self.assertIn("ambíguo", evidence_items[0].reason)

    def test_zero_budget_keeps_board_but_sends_no_memory(self) -> None:
        agent = self.make_agent()
        _, query, recent = TURNS[0]
        base = build_base_messages(
            str(self.config.get("instructions", "")),
            tuple(RecentTurn(text=text) for text in recent),
            query,
        )
        base_tokens = agent.counter.count_messages(base)
        window = base_tokens + self.settings.output_reserve
        turn = agent.before_response(
            "c1",
            query,
            recent,
            window=window,
            output_reserve=self.settings.output_reserve,
        )
        d = turn.budget.diagnostics
        self.assertEqual(d.memory_budget, 0)
        self.assertEqual(d.status, "no_budget")
        self.assertEqual(d.selected_ids, ())
        self.assertEqual(d.memory_tokens, 0)
        self.assertEqual(d.used_tokens, base_tokens)
        self.assertEqual(set(agent.board_for("c1").ids()), {"m1", "m2"})
        system = next(m for m in turn.messages if m["role"] == "system")
        self.assertNotIn("MEMÓRIA", system["content"])

    def test_conversations_are_isolated(self) -> None:
        agent = self.make_agent()
        c1_turn = agent.before_response("c1", *TURNS[0][1:])
        c2_turn = agent.before_response(
            "c2", "Por que comprei a bicicleta?", ("Falamos de bicicleta.",)
        )
        c1_ids = set(agent.board_for("c1").ids())
        c2_ids = set(agent.board_for("c2").ids())
        self.assertEqual(c1_ids, {"m1", "m2"})
        self.assertEqual(c2_ids, {"p1", "p2"})
        self.assertFalse(c1_ids & c2_ids)
        self.assertEqual(c1_turn.budget.diagnostics.selected_ids, ("m2",))
        self.assertEqual(c2_turn.budget.diagnostics.selected_ids, ("p2",))
        system = next(m for m in c2_turn.messages if m["role"] == "system")
        self.assertIn("id=p2", system["content"])
        self.assertNotIn("id=m2", system["content"])

    def test_left_items_keep_original_message_and_can_return(self) -> None:
        agent = self.make_agent()
        turns = self.run_sequence(agent)
        self.assertEqual(agent.store_size, len(self.messages))
        self.assertEqual(agent.message("m2").text, "Porque eu já tinha comprado a licença.")
        self.assertIn("m2", agent.store_ids)
        self.assertIn("m2", turns[0].delta.entered)
        self.assertIn("m2", turns[1].delta.left)
        self.assertIn("m2", turns[3].delta.entered)

    def test_original_is_preserved_on_exit_and_recovered(self) -> None:
        agent = self.make_agent()
        turns = self.run_sequence(agent)
        second = turns[1]
        self.assertIn("m2", second.delta.left)
        self.assertEqual(second.delta.left_reasons["m2"], "no_focus_evidence")
        self.assertNotIn("m2", [item.message_id for item in second.board])
        self.assertEqual(agent.message("m2").text, "Porque eu já tinha comprado a licença.")
        self.assertIn("m2", agent.store_ids)
        fourth = turns[3]
        self.assertIn("m2", [item.message_id for item in fourth.board])
        self.assertIn("m2", fourth.budget.diagnostics.selected_ids)

    def test_recent_history_is_reused_without_double_count(self) -> None:
        agent = self.make_agent()
        query, recent = TURNS[0][1], TURNS[0][2]
        m1 = agent.message("m1")
        turn = agent.before_response(
            "c1",
            query,
            (
                RecentTurn(text=m1.text, id="m1", author="user", position=1),
            )
            + tuple(RecentTurn(text=text) for text in recent),
        )
        d = turn.budget.diagnostics
        self.assertTrue(d.seed_in_base)
        self.assertFalse(d.seed_included)
        self.assertEqual(d.seed_tokens, 0)
        self.assertEqual(d.selected_ids, ("m2",))
        system = next(m for m in turn.messages if m["role"] == "system")
        self.assertNotIn("[semente] id=m1", system["content"])
        self.assertIn("id=m2", system["content"])
        self.assertTrue(any("[m1]" in m["content"] for m in turn.messages))

    def test_budget_counts_the_chat_payload(self) -> None:
        agent = self.make_agent()
        turn = agent.before_response("c1", *TURNS[0][1:])
        d = turn.budget.diagnostics
        self.assertEqual(d.context_format, "chat-messages")
        self.assertTrue(d.counter_simulated)
        self.assertEqual(
            d.used_tokens, EstimatedChatCounter().count_messages(turn.messages)
        )
        self.assertEqual(d.used_tokens - d.base_tokens, d.memory_tokens)
        self.assertLessEqual(d.total_with_reserve, d.window)
        self.assertEqual(d.remaining_tokens, d.window - d.output_reserve - d.used_tokens)

    def test_diagnostics_keep_theme_scores_and_reasons(self) -> None:
        agent = self.make_agent()
        turns = self.run_sequence(agent)
        first = turns[0]
        self.assertEqual(first.themes.dominant, "serum")
        self.assertTrue(first.themes.themes)
        self.assertTrue(all(0.0 <= s.weight <= 1.0 for s in first.themes.themes))
        m2 = next(d for d in first.decisions if d.message_id == "m2")
        self.assertEqual(m2.role, "evidência")
        self.assertAlmostEqual(m2.attention, 0.900, places=6)
        self.assertAlmostEqual(m2.base_attention, 0.900, places=6)
        self.assertAlmostEqual(m2.evidence_focus or 0.0, 1.0)
        self.assertIn("serum", m2.themes)
        self.assertTrue(m2.sent)
        self.assertEqual(m2.budget_reason, "selected")

        third = turns[2]
        left = next(d for d in third.decisions if d.message_id == "m1")
        self.assertEqual(left.action, "left")
        self.assertEqual(left.reason, "outside_radius")

        sixth = turns[5]
        self.assertTrue(sixth.themes.ambiguous)
        self.assertIn("pack", sixth.themes.weights())
        self.assertIn("bass", sixth.themes.weights())
        self.assertIn("pesos próximos", sixth.themes.ambiguity_reason or "")


if __name__ == "__main__":
    unittest.main()
