"""Auditoria de contexto: estrutura e sinais verificáveis sem rede."""

from __future__ import annotations

import unittest

from sinapse.context_audit import CASES, CONDITIONS, run_audit


class ContextAuditTest(unittest.TestCase):
    def test_audit_reports_baseline_targets(self) -> None:
        result = run_audit()
        self.assertEqual(result["state_changes"], 0)
        self.assertEqual(set(result["summary"]), set(CONDITIONS))
        self.assertEqual(result["client"], "simulated")

        current = result["summary"]["current"]
        self.assertGreaterEqual(current["coverage_required_avg"], 0.65)
        self.assertLessEqual(current["missing_total"], 6)
        self.assertLessEqual(current["irrelevant_total"], 9)
        # O conjunto estendido tem 26 casos; os 16/18 comuns ficam abaixo de 491.
        self.assertLessEqual(current["avg_tokens"], 505)
        self.assertEqual(current["duplicate_cases"], 0)
        self.assertEqual(current["leak_cases"], 0)

        semantic = result["summary"]["semantic"]
        self.assertGreaterEqual(semantic["coverage_required_avg"], 0.90)
        self.assertEqual(semantic["irrelevant_total"], 0)
        self.assertEqual(semantic["leak_cases"], 0)
        self.assertEqual(semantic["duplicate_cases"], 0)

        by_id = {entry["id"]: entry for entry in result["cases"]}
        self.assertFalse(by_id["casual"]["conditions"]["current"]["duplicate_query"])
        self.assertEqual(by_id["resume"]["conditions"]["current"]["missing"], [])
        self.assertEqual(by_id["corrected_login"]["conditions"]["current"]["missing"], [])

        credentials = by_id["credentials"]["conditions"]["current"]
        self.assertEqual(credentials["leaks"], [])
        self.assertFalse(
            any(
                "sk-auditoria-ficticia" in str(message.get("content", ""))
                for message in credentials["messages"]
            )
        )

        budget = by_id["budget_pressure"]["conditions"]["current"]
        self.assertEqual(budget["status"], "error")
        self.assertIn("Contexto mínimo excede", budget["error"])
        self.assertEqual(budget["tokens"], 0)

        sent = {
            case: by_id[case]["conditions"]["current"]["sent_ids"]
            for case in ("goal", "corrected_decision", "two_part", "cpu_27", "cpu_30")
        }
        self.assertIn("m1", sent["goal"])
        self.assertIn("m4", sent["corrected_decision"])
        self.assertIn("m4", sent["two_part"])
        self.assertIn("m6", sent["cpu_27"])
        self.assertIn("m11", sent["cpu_27"])
        self.assertIn("m6", sent["cpu_30"])
        self.assertIn("m14", sent["cpu_30"])

        historical = by_id["historical_vocabulary"]["conditions"]["current"]
        self.assertIn("m2", historical["sent_ids"])
        changed = by_id["what_changed"]["conditions"]["current"]
        self.assertIn("m1", changed["sent_ids"])
        self.assertIn("m2", changed["sent_ids"])
        partial = by_id["two_part_partial"]["conditions"]["current"]
        self.assertIn("m2", partial["sent_ids"])
        self.assertIn("m4", partial["sent_ids"])
        unavailable = by_id["unavailable_source"]["conditions"]["current"]
        self.assertEqual(unavailable["missing"], [])
        self.assertTrue(unavailable["gaps"])

        semantic_vocabulary = by_id["semantic_vocabulary"]["conditions"]["semantic"]
        self.assertIn("m2", semantic_vocabulary["sent_ids"])
        self.assertNotIn("m1", semantic_vocabulary["sent_ids"])
        semantic_revoked = by_id["semantic_revoked"]["conditions"]["semantic"]
        self.assertIn("m2", semantic_revoked["sent_ids"])
        self.assertNotIn("m1", semantic_revoked["sent_ids"])
        semantic_gap = by_id["semantic_unavailable"]["conditions"]["semantic"]
        self.assertTrue(semantic_gap["gaps"])
        self.assertEqual(len(CASES), 26)


if __name__ == "__main__":
    unittest.main()
