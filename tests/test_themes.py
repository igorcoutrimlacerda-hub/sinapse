"""Testes da atenção temática: menção, decaimento, reativação e ambiguidade."""

from __future__ import annotations

import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.models import load_messages
from sinapse.themes import analyze_turn, theme_affinity, themes_for_message

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_conversation.json"
CONFIG = ROOT / "data" / "config.json"

SERUM_BOARD = [
    "Escolhi o Serum 2 para fazer o bass.",
    "Porque eu já tinha comprado a licença.",
]


class ThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.by_id = {m.id: m for m in load_messages(DATA)}

    def test_explicit_mentions_win_and_multiple_themes_coexist(self) -> None:
        report = analyze_turn("Como eu divulguei o pack do bass?")
        weights = report.weights()
        self.assertEqual(report.requested_dimension, "como")
        self.assertTrue(report.dimension_explicit)
        self.assertAlmostEqual(weights["pack"], 1.0)
        self.assertAlmostEqual(weights["bass"], 1.0)
        self.assertIn("pack", report.explicit_themes)
        self.assertIn("bass", report.explicit_themes)
        self.assertFalse(report.ambiguous)

    def test_explicit_mention_beats_previous_theme(self) -> None:
        report = analyze_turn(
            "Passei a tarde jogando xadrez.",
            previous_weights={"serum": 1.0, "bass": 0.7},
            board_texts=SERUM_BOARD,
        )
        self.assertEqual(report.dominant, "xadrez")
        self.assertAlmostEqual(report.weights()["xadrez"], 1.0)
        self.assertLess(report.weights()["serum"], 1.0)

    def test_decay_and_history_support(self) -> None:
        first = analyze_turn(
            "Por que eu escolhi o preset Serum 2?",
            ["Falamos do Serum 2 e do bass."],
        )
        self.assertAlmostEqual(first.weights()["serum"], 1.0)
        self.assertAlmostEqual(first.weights()["bass"], 0.4)

        second = analyze_turn(
            "E como eu testei?",
            ["Por que eu escolhi o preset Serum 2?"],
            first.weights(),
            SERUM_BOARD,
        )
        self.assertEqual(second.dominant, "serum")
        self.assertAlmostEqual(second.weights()["bass"], 0.727, places=3)
        self.assertLess(second.weights()["bass"], second.weights()["serum"])

    def test_reactivation_by_new_mention(self) -> None:
        serum = analyze_turn("Por que escolhi o preset Serum 2?")
        xadrez = analyze_turn(
            "Passei a tarde jogando xadrez.",
            ["Por que escolhi o preset Serum 2?"],
            serum.weights(),
            SERUM_BOARD,
        )
        back = analyze_turn(
            "Voltando ao preset: por que eu escolhi ele?",
            ["Passei a tarde jogando xadrez."],
            xadrez.weights(),
            ["Passei a tarde jogando xadrez."],
        )
        self.assertEqual(back.dominant, "serum")
        self.assertAlmostEqual(back.weights()["serum"], 1.0)
        self.assertAlmostEqual(back.weights()["xadrez"], 0.55, places=3)
        self.assertIn("serum", back.explicit_themes)

    def test_ambiguity_when_weights_are_close_without_explicit_mention(self) -> None:
        report = analyze_turn(
            "E aquilo que eu preparei?",
            ["Como eu divulguei o pack do bass?"],
            {"pack": 1.0, "bass": 1.0, "serum": 0.55},
            ["Preparei o pack do bass para divulgar."],
        )
        self.assertTrue(report.ambiguous)
        self.assertIsNotNone(report.ambiguity_reason)
        self.assertIn("pack", report.ambiguity_reason or "")
        self.assertIn("bass", report.ambiguity_reason or "")
        self.assertAlmostEqual(report.weights()["pack"], 1.0)
        self.assertAlmostEqual(report.weights()["bass"], 1.0)

    def test_explicit_mention_does_not_flag_ambiguity(self) -> None:
        report = analyze_turn(
            "E o pack do bass?",
            previous_weights={"pack": 1.0, "bass": 1.0},
        )
        self.assertFalse(report.ambiguous)
        self.assertIn(report.dominant, {"bass", "pack"})

    def test_reply_inherits_theme_from_parent(self) -> None:
        m2 = self.by_id["m2"]
        weights = {"serum": 1.0, "bass": 1.0}
        self.assertAlmostEqual(theme_affinity(m2, weights, self.by_id), 1.0)
        self.assertEqual(
            set(themes_for_message(m2, weights, self.by_id)), {"serum", "bass"}
        )
        # Sem o vínculo, a licença não tem termo temático próprio.
        self.assertEqual(theme_affinity(m2, weights), 0.0)

    def test_report_has_observable_signals(self) -> None:
        report = analyze_turn("Por que eu escolhi o preset Serum 2?")
        self.assertEqual(report.requested_dimension, "por_que")
        self.assertTrue(any("menção explícita" in s for s in report.signals))
        self.assertTrue(any("dimensão pedida" in s for s in report.signals))
        self.assertEqual(report.as_dict()["dominant"], "serum")


if __name__ == "__main__":
    unittest.main()
