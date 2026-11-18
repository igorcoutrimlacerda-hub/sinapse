"""Testes do cenário de retomada de projeto e do rastreamento da pendência."""

from __future__ import annotations

import unittest

from sinapse.config import load_config
from sinapse.deepseek import load_deepseek_settings
from sinapse.resumption import cases, evaluate_answer, run_resumption_benchmark


class ResumptionBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.settings = load_deepseek_settings(cls.config)
        cls.result = run_resumption_benchmark(cls.config, cls.settings)

    def _row(self, case: str, condition: str) -> dict:
        return next(
            row
            for row in self.result["rows"]
            if row["case"] == case and row["condition"] == condition
        )

    def test_cases_and_conditions(self) -> None:
        self.assertEqual(self.result["cases"], 10)
        self.assertEqual(len(self.result["rows"]), 30)
        self.assertEqual(
            {row["condition"] for row in self.result["rows"]},
            {"history", "sinapse", "sinapse_multipurpose"},
        )
        self.assertEqual(self.result["mode"], "simulated")
        self.assertFalse(self.result["real_evaluation"]["available"])
        self.assertFalse(self.result["real_pairing"]["available"])

    def test_history_cannot_resume_without_memory(self) -> None:
        history = self.result["aggregates"]["history"]
        self.assertEqual(history["avg_score"], 0.0)
        self.assertEqual(history["evidence_coverage"], 0.0)
        self.assertEqual(history["memory_tokens_avg"], 0.0)

    def test_sinapse_covers_most_elements(self) -> None:
        sinapse = self.result["aggregates"]["sinapse"]
        self.assertEqual(sinapse["decision_ok"], sinapse["decision_total"])
        self.assertEqual(sinapse["reason_ok"], sinapse["reason_total"])
        self.assertGreaterEqual(sinapse["completed_ok"], 6)
        self.assertGreaterEqual(sinapse["pending_ok"], 8)
        self.assertGreater(sinapse["coverage"], 0.9)
        self.assertGreaterEqual(sinapse["references_all"], 5)
        self.assertGreaterEqual(sinapse["reference_support_all"], 5)
        self.assertEqual(sinapse["completed_repetitions"], 0)
        self.assertEqual(sinapse["unsupported_claims"], 0)
        self.assertLessEqual(sinapse["memory_tokens_avg"], 160)

    def test_multipurpose_closes_pending_gap(self) -> None:
        multipurpose = self.result["aggregates"]["sinapse_multipurpose"]
        self.assertEqual(multipurpose["coverage"], 1.0)
        self.assertEqual(
            multipurpose["decision_ok"], multipurpose["decision_total"]
        )
        self.assertEqual(
            multipurpose["reason_ok"], multipurpose["reason_total"]
        )
        self.assertEqual(
            multipurpose["completed_ok"], multipurpose["completed_total"]
        )
        self.assertEqual(
            multipurpose["pending_ok"], multipurpose["pending_total"]
        )
        self.assertEqual(multipurpose["references_all"], 9)
        self.assertEqual(multipurpose["reference_support_all"], 10)
        self.assertEqual(multipurpose["conditions_ok"], 1)
        self.assertEqual(multipurpose["conditions_omitted"], 0)
        self.assertEqual(multipurpose["old_tasks_as_current"], 0)
        self.assertEqual(multipurpose["wrong_project"], 0)
        self.assertEqual(multipurpose["errors"], 0)
        self.assertEqual(multipurpose["completed_repetitions"], 0)
        self.assertEqual(multipurpose["unsupported_claims"], 0)
        self.assertLessEqual(multipurpose["memory_tokens_avg"], 160)
        history = self.result["aggregates"]["history"]
        self.assertGreater(
            multipurpose["prompt_tokens_total"],
            history["prompt_tokens_total"],
        )

    def test_no_regression_against_known_cases(self) -> None:
        self.assertEqual(self.result["multipurpose_worse"], 0)
        self.assertEqual(self.result["multipurpose_better"], 3)
        self.assertGreaterEqual(self.result["sinapse_better_history"], 9)
        regression = self.result["aggregates_regression"]
        baseline = regression["sinapse"]
        targeted = regression["sinapse_multipurpose"]
        baseline_total = sum(
            baseline[f"{name}_ok"] for name in ("decision", "reason", "completed", "pending")
        )
        targeted_total = sum(
            targeted[f"{name}_ok"] for name in ("decision", "reason", "completed", "pending")
        )
        self.assertEqual(baseline_total, 18)
        self.assertEqual(targeted_total, 20)
        new = self.result["aggregates_new"]["sinapse_multipurpose"]
        new_total = sum(
            new[f"{name}_ok"] for name in ("decision", "reason", "completed", "pending")
        )
        expected_total = sum(
            new[f"{name}_total"] for name in ("decision", "reason", "completed", "pending")
        )
        self.assertEqual(new_total, expected_total)
        self.assertEqual(new["conditions_ok"], new["conditions_total"])

    def test_retomada_geral_recovers_pending_only_with_multipurpose(self) -> None:
        baseline = self._row("retomada_geral", "sinapse")
        targeted = self._row("retomada_geral", "sinapse_multipurpose")
        self.assertFalse(baseline["element_checks"]["pending"])
        self.assertTrue(targeted["element_checks"]["pending"])
        self.assertIn("pr7", targeted["selected_ids"])
        self.assertTrue(targeted["references"]["pending"])

    def test_trace_shows_pending_outside_radius(self) -> None:
        baseline = next(
            trace
            for trace in self.result["traces"]
            if trace["case"] == "retomada_geral"
            and trace["condition"] == "sinapse"
        )
        pending = {
            item["id"]: item
            for item in baseline["candidates"]
            if item["id"] in {"pr6", "pr7"}
        }
        self.assertTrue(pending)
        self.assertTrue(
            all(not item["within_radius"] for item in pending.values())
        )
        self.assertTrue(
            all(item["attention"] == 0.0 for item in pending.values())
        )

    def test_completed_pending_does_not_return_as_pending(self) -> None:
        row = self._row("pendencia_concluida", "sinapse_multipurpose")
        self.assertTrue(all(row["element_checks"].values()))
        self.assertIn("sr5", row["selected_ids"])
        self.assertIn("sr6", row["selected_ids"])
        self.assertNotIn("sr4", row["selected_ids"])
        self.assertTrue(row["references"]["completed"])
        self.assertTrue(row["references"]["pending"])

    def test_substituted_pending_is_not_reintroduced(self) -> None:
        row = self._row("pendencia_substituida", "sinapse_multipurpose")
        self.assertTrue(all(row["element_checks"].values()))
        self.assertTrue({"lt5", "lt6"} & set(row["selected_ids"]))
        self.assertNotIn("lt4", row["selected_ids"])

    def test_project_without_next_step_is_acknowledged(self) -> None:
        row = self._row("sem_pendencia", "sinapse_multipurpose")
        self.assertTrue(all(row["element_checks"].values()))
        self.assertIn("vl4", row["selected_ids"])
        self.assertEqual(row["references"]["pending"], True)

    def test_conditional_next_step_keeps_condition(self) -> None:
        row = self._row("proximo_condicionado", "sinapse_multipurpose")
        self.assertTrue(all(row["element_checks"].values()))
        self.assertEqual(row["checks"]["conditions_omitted"], [])
        self.assertTrue(row["references"]["condition"])
        self.assertTrue(row["reference_support"]["condition"])
        self.assertIn("dn4", row["selected_ids"])
        self.assertEqual(row["errors"], [])

    def test_interleaved_projects_do_not_mix(self) -> None:
        north = self._row("intercalado_norte", "sinapse_multipurpose")
        south = self._row("intercalado_sul", "sinapse_multipurpose")
        self.assertTrue(all(north["element_checks"].values()))
        self.assertTrue(all(south["element_checks"].values()))
        self.assertEqual(north["wrong_project"], [])
        self.assertEqual(south["wrong_project"], [])
        self.assertTrue({"n2", "n3", "n4"} <= set(north["selected_ids"]))
        self.assertTrue({"s2", "s3", "s4"} <= set(south["selected_ids"]))

    def test_old_task_does_not_return_as_current(self) -> None:
        row = self._row("pendencia_substituida_costa", "sinapse_multipurpose")
        self.assertTrue(all(row["element_checks"].values()))
        self.assertNotIn("cs4", row["selected_ids"])
        self.assertEqual(row["old_tasks_as_current"], [])

    def test_references_support_their_elements(self) -> None:
        for row in self.result["rows"]:
            if row["condition"] != "sinapse_multipurpose":
                continue
            for name in row["checks"]["components"] + (
                ["condition"] if "condition" in row["references"] else []
            ):
                self.assertTrue(
                    row["reference_support"].get(name, False), row["case"]
                )
            self.assertEqual(row["reference_support_failures"], [], row["case"])

    def test_rubric_full_answer(self) -> None:
        case = cases()[0]
        score, checks = evaluate_answer(case, case.reference_answer)
        self.assertEqual(score, 1.0)
        self.assertTrue(checks["decision"])
        self.assertTrue(checks["reason"])
        self.assertTrue(checks["completed"])
        self.assertTrue(checks["pending"])
        self.assertEqual(checks["completed_repetition"], [])
        self.assertEqual(checks["unsupported_claims"], [])

    def test_rubric_flags_repeated_completed_step(self) -> None:
        case = cases()[0]
        score, checks = evaluate_answer(
            case, "Sugiro repetir o teste do refrão para confirmar."
        )
        self.assertIn(
            "repetir o teste do refrão", checks["completed_repetition"]
        )
        self.assertFalse(checks["pending"])
        self.assertEqual(score, 0.0)

    def test_rubric_flags_unsupported_claims(self) -> None:
        case = cases()[0]
        score, checks = evaluate_answer(
            case, "O prazo é sexta e o cliente aprovou o Pigments."
        )
        self.assertEqual(
            set(checks["unsupported_claims"]), {"prazo", "cliente"}
        )
        self.assertEqual(score, 0.0)

    def test_rubric_rejects_old_decision_as_current(self) -> None:
        case = next(item for item in cases() if item.id == "decisao_vigente")
        _, checks = evaluate_answer(case, "Uso o Aurora no bass.")
        self.assertFalse(checks["decision"])


if __name__ == "__main__":
    unittest.main()
