"""Calibração da rubrica v2 e verificação dos artefatos de comparação."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sinapse.resumption_rubric_v2 import calibrate


class RubricV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = calibrate()
        cls.synthetic = {item["id"]: item for item in cls.result["synthetic"]}

    def test_calibration_has_no_false_positives_or_negatives(self) -> None:
        for name, item in self.result["criteria"].items():
            self.assertEqual(item["fp"], 0, name)
            self.assertEqual(item["fn"], 0, name)

    def test_expected_true_positives(self) -> None:
        criteria = self.result["criteria"]
        self.assertEqual(criteria["decision"]["tp"], 18)
        self.assertEqual(criteria["reason"]["tp"], 20)
        self.assertEqual(criteria["completed"]["tp"], 14)
        self.assertEqual(criteria["pending"]["tp"], 16)
        self.assertEqual(criteria["condition"]["tp"], 2)

    def test_synthetic_cases_match_gold(self) -> None:
        for item in self.result["synthetic"]:
            for name, gold in item["gold"].items():
                self.assertEqual(item["v2"][name], bool(gold), item["id"])

    def test_old_decision_is_not_current(self) -> None:
        item = self.synthetic["antiga_como_atual"]
        self.assertFalse(item["v2"]["decision"])
        self.assertTrue(item["v2"]["old_or_ambiguous"])

    def test_reformulated_next_step_is_rejected(self) -> None:
        self.assertFalse(self.synthetic["proximo_reformulado"]["v2"]["pending"])
        self.assertTrue(self.synthetic["proximo_correto"]["v2"]["pending"])

    def test_incomplete_condition_is_rejected(self) -> None:
        self.assertFalse(self.synthetic["condicao_incompleta"]["v2"]["condition"])
        self.assertTrue(
            self.synthetic["condicao_dois_caminhos"]["v2"]["condition"]
        )

    def test_rp02_analysis_and_instruction_plan(self) -> None:
        analysis = json.loads(
            Path("benchmark_out_resumption_real/rp02_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(analysis["request"]["pr7_in_both_requests"])
        self.assertTrue(analysis["A"]["goal_cpu_preserved"])
        self.assertTrue(analysis["B"]["goal_cpu_preserved"])
        self.assertNotEqual(analysis["A"]["object"], analysis["B"]["object"])
        plan = json.loads(
            Path(
                "benchmark_out_resumption_real/instruction_experiment_plan.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(plan["predicted_real_calls"], 20)
        self.assertEqual(plan["memory_budget_tokens"], 160)
        self.assertEqual(plan["new_rubric_cases_count"], 6)


if __name__ == "__main__":
    unittest.main()
