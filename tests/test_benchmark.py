"""Testes do benchmark: casos, seleção, respostas simuladas e exportação."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sinapse.benchmark import export, load_cases, run_benchmark, score_answer, summarize
from sinapse.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmark" / "cases_eval.json"
CONFIG = ROOT / "data" / "config.json"


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)
        cls.cases = load_cases(CASES)
        cls.by_id = {case.id: case for case in cls.cases}
        cls.subset_ids = (
            "motivo_serum",
            "similar_motivo",
            "restrito_sem_pedido",
            "restrito_com_pedido",
            "historico_repeticao",
            "sem_informacao",
        )
        cls.results = run_benchmark(
            [case for case in cls.cases if case.id in cls.subset_ids], cls.config
        )
        cls.result_by_id = {result.case.id: result for result in cls.results}

    def test_cases_cover_required_shapes(self) -> None:
        kinds = {case.kind for case in self.cases}
        self.assertGreaterEqual(len(self.cases), 12)
        for required in (
            "motivo",
            "procedimento",
            "referencia_curta",
            "retomada",
            "correcao",
            "duas_evidencias",
            "falta_de_informacao",
            "restricao_sem_pedido",
            "restricao_com_pedido",
            "exclusao",
        ):
            self.assertIn(required, kinds)
        for case in self.cases:
            self.assertTrue(case.messages, case.id)
            self.assertTrue(case.question, case.id)

    def test_sinapse_selects_answer_while_lexical_does_not(self) -> None:
        motive = self.result_by_id["motivo_serum"]
        self.assertTrue(motive.outcomes["sinapse"].hit_any)
        self.assertTrue(motive.outcomes["sinapse"].full_coverage)
        self.assertFalse(motive.outcomes["lexical"].hit_any)
        self.assertFalse(motive.outcomes["history"].hit_any)
        self.assertTrue(motive.outcomes["manual"].full_coverage)
        self.assertGreater(motive.outcomes["sinapse"].additional_tokens, 0)
        self.assertEqual(
            self.result_by_id["motivo_serum"].outcomes["sinapse"].answer_score, 1.0
        )

    def test_similar_messages_do_not_displace_the_answer(self) -> None:
        result = self.result_by_id["similar_motivo"]
        sinapse = result.outcomes["sinapse"]
        lexical = result.outcomes["lexical"]
        self.assertTrue(sinapse.full_coverage)
        self.assertIn("m2", sinapse.selected_ids)
        self.assertFalse(set(sinapse.selected_ids) & set(result.case.noise_ids))
        self.assertFalse(lexical.full_coverage)
        self.assertTrue(set(lexical.selected_ids) & set(result.case.noise_ids))
        self.assertLess(sinapse.additional_tokens, 113)

    def test_restriction_keeps_memory_out_and_diagnostics_clean(self) -> None:
        without = self.result_by_id["restrito_sem_pedido"]
        outcome = without.outcomes["sinapse"]
        self.assertFalse(outcome.restricted_violations)
        self.assertNotIn("m2", outcome.selected_ids)
        context = "\n".join(message["content"] for message in outcome.messages)
        self.assertNotIn("licença", context)
        self.assertNotIn("licenca", context)
        self.assertIn(without.case.question, context)

        with_request = self.result_by_id["restrito_com_pedido"]
        released = with_request.outcomes["sinapse"]
        self.assertIn("m2", released.selected_ids)
        self.assertTrue(released.full_coverage)
        self.assertFalse(released.restricted_violations)

    def test_history_reuse_avoids_repetition(self) -> None:
        result = self.result_by_id["historico_repeticao"]
        self.assertTrue(result.outcomes["history"].full_coverage)
        for condition in ("sinapse", "manual", "lexical"):
            outcome = result.outcomes[condition]
            self.assertNotIn("m2", outcome.selected_ids)
            self.assertFalse(outcome.repetition)

    def test_missing_information_case_leaves_memory_empty(self) -> None:
        result = self.result_by_id["sem_informacao"]
        for condition in ("history", "lexical", "sinapse", "manual"):
            outcome = result.outcomes[condition]
            self.assertEqual(outcome.additional_tokens, 0)
            self.assertEqual(outcome.selected_ids, ())
            self.assertFalse(outcome.irrelevant_attached)
        self.assertTrue(result.outcomes["sinapse"].checks["acknowledged_missing"])

    def test_same_ids_have_same_cost(self) -> None:
        motive = self.result_by_id["motivo_serum"]
        self.assertEqual(
            motive.outcomes["sinapse"].selected_ids,
            motive.outcomes["manual"].selected_ids,
        )
        self.assertEqual(
            motive.outcomes["sinapse"].additional_tokens,
            motive.outcomes["manual"].additional_tokens,
        )
        self.assertEqual(motive.outcomes["sinapse"].additional_tokens, 81)

    def test_price_presence_absence_and_other_numbers(self) -> None:
        tuning = {
            case.id: case
            for case in load_cases(ROOT / "benchmark" / "cases_tuning.json")
        }
        price_cases = [
            "tuning_preco_presente_mensagem",
            "tuning_preco_resposta_curta",
            "tuning_numero_quantidade",
            "tuning_numero_duracao",
            "tuning_preco_tema_diferente",
            "tuning_preco_variacao_linguagem",
        ]
        results = {
            result.case.id: result
            for result in run_benchmark(
                [tuning[case_id] for case_id in price_cases], self.config
            )
        }
        present = results["tuning_preco_presente_mensagem"].outcomes["sinapse"]
        self.assertTrue(present.hit_any)
        self.assertIn("m10", present.selected_ids)
        reply = results["tuning_preco_resposta_curta"].outcomes["sinapse"]
        self.assertTrue(reply.hit_any)
        self.assertIn("m10", reply.selected_ids)
        for absent_id in (
            "tuning_numero_quantidade",
            "tuning_numero_duracao",
            "tuning_preco_variacao_linguagem",
        ):
            outcome = results[absent_id].outcomes["sinapse"]
            self.assertEqual(outcome.selected_ids, ())
            self.assertEqual(outcome.additional_tokens, 0)
        bicycle = results["tuning_preco_tema_diferente"].outcomes["sinapse"]
        self.assertIn("p3", bicycle.selected_ids)

    def test_hit_and_full_report_counts_with_denominator(self) -> None:
        summary = summarize(self.results)
        for condition in ("history", "lexical", "sinapse", "manual"):
            item = summary["conditions"][condition]
            self.assertEqual(item["hit_denominator"], 4)
            self.assertLessEqual(item["hit_count"], item["hit_denominator"])
            self.assertAlmostEqual(
                item["hit_rate"], item["hit_count"] / item["hit_denominator"]
            )

    def test_v2_set_is_separate_and_sized_for_four_conditions(self) -> None:
        v2_cases = load_cases(ROOT / "benchmark" / "cases_eval_v2.json")
        self.assertEqual(len(v2_cases), 11)
        kinds = {case.kind for case in v2_cases}
        self.assertIn("falta_de_evidencia", kinds)
        self.assertIn("multiplas_evidencias", kinds)
        self.assertIn("correcao", kinds)
        self.assertIn("respondivel_pelo_historico", kinds)
        self.assertEqual(len(v2_cases) * 4, 44)
        ids = {case.id for case in v2_cases}
        self.assertFalse(ids & {case.id for case in self.cases})

    def test_summary_reports_deltas_and_per_case(self) -> None:
        summary = summarize(self.results)
        self.assertEqual(summary["cases"], len(self.results))
        self.assertIn("sinapse_delta_pp", summary)
        self.assertIn("per_case", summary)
        self.assertGreaterEqual(summary["conditions"]["sinapse"]["hit_rate"], 0.0)
        self.assertLessEqual(summary["conditions"]["sinapse"]["hit_rate"], 1.0)
        for entry in summary["per_case"]:
            self.assertIn("better_than", entry)
            self.assertIn("worse_than", entry)

    def test_score_answer_penalizes_forbidden_and_missing(self) -> None:
        case = self.by_id["sem_informacao"]
        score, checks = score_answer(case, "Não tenho informação sobre o preço.")
        self.assertEqual(score, 1.0)
        self.assertTrue(checks["acknowledged_missing"])
        score, checks = score_answer(case, "O preço foi 200 reais.")
        self.assertLess(score, 1.0)
        self.assertTrue(checks["forbidden_claims_present"])

    def test_export_hides_condition_in_review(self) -> None:
        summary = summarize(self.results)
        with tempfile.TemporaryDirectory() as directory:
            paths = export(self.results, summary, directory)
            review = json.loads(Path(paths["review"]).read_text(encoding="utf-8"))
            self.assertEqual(len(review), len(self.results) * 4)
            for entry in review:
                self.assertNotIn("condition", entry)
                self.assertNotIn("sinapse", json.dumps(entry, ensure_ascii=False))
            results = json.loads(Path(paths["results"]).read_text(encoding="utf-8"))
            self.assertIn("summary", results)
            self.assertIn("cases", results)


if __name__ == "__main__":
    unittest.main()
