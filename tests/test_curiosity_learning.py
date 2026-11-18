"""Testes da curiosidade aprendida: ações, rótulos e comparação."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.curiosity_learning import ACTIONS, cases, run_experiment


class CuriosityLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_cases, cls.eval_cases = cases()
        cls.result = run_experiment(load_config("data/config.json"))

    def test_train_and_eval_are_separate_families(self) -> None:
        train_families = {case.family for case in self.train_cases}
        eval_families = {case.family for case in self.eval_cases}
        self.assertGreaterEqual(len(self.train_cases), 5)
        self.assertGreaterEqual(len(self.eval_cases), 5)
        self.assertEqual(len(self.result["model"]["weights"]), len(ACTIONS))
        self.assertGreater(self.result["training_rows"], 0)

    def test_learned_policy_improves_or_matches_fixed(self) -> None:
        self.assertGreaterEqual(
            self.result["learned"]["full"], self.result["fixed_rules"]["full"]
        )
        self.assertGreaterEqual(
            self.result["fixed_rules"]["full"], self.result["off"]["full"]
        )
        self.assertLessEqual(
            self.result["learned"]["no_gain_searches"],
            self.result["fixed_rules"]["no_gain_searches"],
        )

    def test_changed_actions_are_reported(self) -> None:
        changed = [
            entry
            for entry in self.result["evaluation"]
            if entry["learned_action"] != entry["fixed_action"]
        ]
        self.assertTrue(changed)
        for entry in changed:
            self.assertIn(entry["learned_action"], ACTIONS)
            self.assertIn(entry["fixed_action"], ACTIONS)
            if entry["improved"]:
                self.assertGreater(entry["learned"]["full"], entry["fixed"]["full"])

    def test_sensitivity_is_applied_to_all_actions(self) -> None:
        entry = next(
            item for item in self.result["evaluation"] if item["case"] == "ev_sensitive"
        )
        for key in ("stop", "fixed", "learned"):
            context = " ".join(
                str(message["content"]) for message in entry[key]["messages"]
            )
            self.assertNotIn("id=m2", context)
            self.assertNotIn("licença", context)
            self.assertNotIn("licenca", context)


if __name__ == "__main__":
    unittest.main()
