"""Testes da utilidade marginal: novidade, repetição e parada antecipada."""

from __future__ import annotations

import unittest
from pathlib import Path

from sinapse.benchmark import load_cases
from sinapse.budget import RecentTurn
from sinapse.chat import EstimatedChatCounter, build_base_messages, select_chat_context
from sinapse.config import load_config
from sinapse.memory import recall
from sinapse.utility import estimate_utility

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmark" / "cases_eval.json"
CONFIG = ROOT / "data" / "config.json"


class UtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)
        cls.cases = {case.id: case for case in load_cases(CASES)}

    def test_answer_adds_more_than_similar_subject_messages(self) -> None:
        case = self.cases["similar_motivo"]
        by_id = {message.id: message for message in case.messages}
        recent = ["Falamos do Serum 2 e do bass."]
        answer = estimate_utility(by_id["m2"], 1.0, 0.9, recent, [])
        noise = estimate_utility(by_id["m10"], 1.0, 0.9, recent, [])
        self.assertGreater(answer.gain, noise.gain)
        self.assertGreater(answer.novelty, noise.novelty)
        self.assertLess(noise.novelty, 0.55)
        self.assertEqual(answer.reason, "novo detalhe para o pedido")

    def test_repeated_assertion_has_low_gain(self) -> None:
        case = self.cases["motivo_serum"]
        m2 = next(message for message in case.messages if message.id == "m2")
        repeated = estimate_utility(
            m2, 1.0, 0.9, ["Porque eu já tinha comprado a licença."], []
        )
        self.assertTrue(repeated.same_assertion)
        self.assertLess(repeated.gain, 0.2)
        self.assertIn("mesma afirmação", repeated.reason)

    def test_different_number_preserves_novelty(self) -> None:
        case = self.cases["motivo_serum"]
        m2 = next(message for message in case.messages if message.id == "m2")
        changed = estimate_utility(
            m2, 1.0, 0.9, ["Porque eu já tinha comprado a licença por 100 reais."], []
        )
        self.assertGreaterEqual(changed.novelty, 0.65)
        self.assertIn("diferença", changed.reason)

    def test_selection_stops_early_when_noise_adds_little(self) -> None:
        case = self.cases["similar_motivo"]
        by_id = {message.id: message for message in case.messages}
        config = self.config
        recent = (RecentTurn(text="Falamos do Serum 2 e do bass."),)
        seed_result = recall(
            case.messages,
            case.question,
            [turn.text for turn in recent],
            config,
            conversation_id=case.conversation_id,
        )
        candidates = []
        for message_id in ("m2", "m10", "m11"):
            message = by_id[message_id]
            evidence = {dimension: None for dimension in ("o_que", "quem", "onde", "quando", "como", "por_que", "para_que")}
            evidence["por_que"] = 1.0
            candidates.append(
                type(seed_result.candidates[0])(
                    message=message,
                    distance=0.117,
                    association=0.9,
                    association_content=0.0,
                    association_link=0.9,
                    association_source="link",
                    evidence=evidence,
                    attention=0.9,
                    within_radius=True,
                )
            )
        counter = EstimatedChatCounter()
        base = build_base_messages(
            str(config.get("instructions", "")), recent, case.question
        )
        base_tokens = counter.count_messages(base)
        result = select_chat_context(
            instructions=str(config.get("instructions", "")),
            recent_turns=recent,
            current_turn=case.question,
            seed=seed_result.seed,
            focus="por_que",
            candidates=candidates,
            window=base_tokens + 4096 + case.memory_token_limit,
            output_reserve=4096,
            counter=counter,
            min_gain=0.05,
            min_novelty=0.55,
        )
        self.assertEqual(result.diagnostics.selected_ids, ("m2",))
        self.assertTrue(result.diagnostics.stopped_early)
        decisions = {decision.id: decision for decision in result.diagnostics.decisions}
        self.assertIsNotNone(decisions["m10"].marginal_gain)
        self.assertIn(decisions["m10"].reason, {"below_marginal_gain", "top_k_full"})

    def test_evidence_already_in_history_is_not_repeated(self) -> None:
        case = self.cases["historico_repeticao"]
        m2 = next(message for message in case.messages if message.id == "m2")
        config = self.config
        recent = (RecentTurn(text=m2.text, id="m2", author="user"),)
        seed_result = recall(
            case.messages,
            case.question,
            [turn.text for turn in recent],
            config,
            conversation_id=case.conversation_id,
        )
        evidence = {dimension: None for dimension in ("o_que", "quem", "onde", "quando", "como", "por_que", "para_que")}
        evidence["por_que"] = 1.0
        from sinapse.models import Candidate

        candidate = Candidate(
            message=m2,
            distance=0.117,
            association=0.9,
            association_content=0.0,
            association_link=0.9,
            association_source="link",
            evidence=evidence,
            attention=0.9,
            within_radius=True,
        )
        counter = EstimatedChatCounter()
        base = build_base_messages(
            str(config.get("instructions", "")), recent, case.question
        )
        base_tokens = counter.count_messages(base)
        result = select_chat_context(
            instructions=str(config.get("instructions", "")),
            recent_turns=recent,
            current_turn=case.question,
            seed=seed_result.seed,
            focus="por_que",
            candidates=[candidate],
            window=base_tokens + 4096 + 140,
            output_reserve=4096,
            counter=counter,
        )
        self.assertEqual(result.diagnostics.selected_ids, ())
        decision = result.diagnostics.decisions[0]
        self.assertEqual(decision.reason, "already_in_base")


if __name__ == "__main__":
    unittest.main()
