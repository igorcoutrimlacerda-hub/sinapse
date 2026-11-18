"""Testes de sensibilidade: declaração, inferência, escopo, liberação e exclusão."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sinapse.config import load_config
from sinapse.models import load_messages
from sinapse.sensitivity import (
    ACTION_DELETE,
    ACTION_RESTRICT,
    SCOPE_DETAIL,
    SCOPE_MESSAGE,
    SCOPE_SUBJECT,
    SOURCE_DECLARED,
    SOURCE_INFERENCE,
    SensitivityPolicy,
)
from sinapse.whiteboard import MemoryAgent

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "agent_conversation.json"
CONFIG = ROOT / "data" / "config.json"


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages = load_messages(DATA)
        cls.by_id = {message.id: message for message in cls.messages}
        cls.config = load_config(CONFIG)

    def test_declared_restriction_scope_and_release(self) -> None:
        policy = SensitivityPolicy()
        events = policy.declare(
            "c1", "Não quero mais falar sobre a licença do Serum 2.", 1
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, SOURCE_DECLARED)
        self.assertEqual(events[0].scope, SCOPE_DETAIL)
        self.assertEqual(events[0].action, ACTION_RESTRICT)

        blocked = policy.evaluate(
            "c1", "Por que eu escolhi o preset Serum 2?", self.messages, 2
        )
        decisions = {d.message_id: d for d in blocked.decisions}
        self.assertFalse(decisions["m2"].allowed)
        self.assertEqual(decisions["m2"].reason, "detalhe restrito")
        self.assertNotIn("m2", blocked.released_ids)

        # Referência ambígua não libera.
        ambiguous = policy.evaluate("c1", "E aquilo?", self.messages, 3)
        self.assertFalse(
            next(d for d in ambiguous.decisions if d.message_id == "m2").allowed
        )

        # Pedido claro e específico libera só aquele detalhe naquele turno.
        clear = policy.evaluate(
            "c1", "Lembra do detalhe da licença? Pode me contar.", self.messages, 4
        )
        released = next(d for d in clear.decisions if d.message_id == "m2")
        self.assertTrue(released.allowed)
        self.assertTrue(released.released)
        again = policy.evaluate(
            "c1", "Por que eu escolhi o preset Serum 2?", self.messages, 5
        )
        self.assertFalse(
            next(d for d in again.decisions if d.message_id == "m2").allowed
        )

    def test_scope_subject_message_and_detail(self) -> None:
        policy = SensitivityPolicy()
        policy.declare("c1", "Não fale sobre xadrez.", 1)
        policy.declare("c1", "Não mencione a mensagem m6.", 1)
        policy.declare("c1", "Restrinja o detalhe dos envelopes.", 1)
        report = policy.evaluate("c1", "Como foi o jogo?", self.messages, 2)
        decisions = {d.message_id: d for d in report.decisions}
        self.assertFalse(decisions["m4"].allowed)
        self.assertEqual(decisions["m4"].scope, SCOPE_SUBJECT)
        self.assertFalse(decisions["m6"].allowed)
        self.assertEqual(decisions["m6"].scope, SCOPE_MESSAGE)
        self.assertFalse(decisions["m3"].allowed)
        self.assertEqual(decisions["m3"].scope, SCOPE_DETAIL)

    def test_inference_is_separate_and_explicit_wins(self) -> None:
        policy = SensitivityPolicy()
        inferred = policy.infer(
            "c1", "Esse assunto de xadrez me deixa desconfortável.", 1
        )
        self.assertEqual(inferred[0].source, SOURCE_INFERENCE)
        report = policy.evaluate("c1", "E aquilo?", self.messages, 2)
        self.assertFalse(
            next(d for d in report.decisions if d.message_id == "m4").allowed
        )

        declared = policy.declare("c1", "Pode falar sobre xadrez de novo.", 3)
        self.assertEqual(declared[0].source, SOURCE_DECLARED)
        self.assertEqual(declared[0].operation, "updated")
        allowed = policy.evaluate("c1", "E aquilo?", self.messages, 4)
        self.assertTrue(
            next(d for d in allowed.decisions if d.message_id == "m4").allowed
        )

    def test_explicit_restriction_beats_inference(self) -> None:
        policy = SensitivityPolicy()
        policy.infer("c1", "O assunto de xadrez me deixa desconfortável.", 1)
        policy.declare("c1", "Não quero falar de xadrez.", 2)
        event = policy.declare("c1", "Pode falar de xadrez.", 3)
        self.assertEqual(event[0].source, SOURCE_DECLARED)
        report = policy.evaluate("c1", "E aquilo?", self.messages, 4)
        self.assertTrue(
            next(d for d in report.decisions if d.message_id == "m4").allowed
        )

    def test_deletion_is_distinct_and_permanent(self) -> None:
        policy = SensitivityPolicy()
        policy.declare("c1", "Apague a mensagem m7.", 1)
        report = policy.evaluate("c1", "Lembra do pack? Pode contar.", self.messages, 2)
        deleted = next(d for d in report.decisions if d.message_id == "m7")
        self.assertFalse(deleted.allowed)
        self.assertEqual(deleted.action, ACTION_DELETE)
        clear = policy.evaluate(
            "c1", "Lembra da mensagem m7? Pode mostrar o detalhe?", self.messages, 3
        )
        self.assertFalse(
            next(d for d in clear.decisions if d.message_id == "m7").allowed
        )

    def test_summary_and_history_are_filtered(self) -> None:
        agent = MemoryAgent(self.messages, self.config)
        agent.declare_preference("c1", "Não quero mais falar sobre a licença do Serum 2.")
        summary = agent.add_summary(
            "c1",
            "Resumo: escolhi o Serum 2 porque já tinha comprado a licença.",
            ("m1", "m2"),
        )
        turn = agent.before_response(
            "c1",
            "Por que eu escolhi o preset Serum 2?",
            (
                {"text": "Porque eu já tinha comprado a licença.", "id": "m2"},
                "Falamos do Serum 2 e do bass.",
            ),
        )
        self.assertNotIn("m2", turn.budget.diagnostics.selected_ids)
        context = "\n".join(message["content"] for message in turn.messages)
        self.assertNotIn("licença", context)
        self.assertNotIn("licenca", context)
        summary_decision = next(
            decision for decision in turn.sensitivity.summaries
            if decision.summary_id == summary.id
        )
        self.assertFalse(summary_decision.allowed)
        self.assertIn("m2", turn.sensitivity.omitted_history)

    def test_diagnostics_do_not_reproduce_restricted_text(self) -> None:
        agent = MemoryAgent(self.messages, self.config)
        agent.declare_preference("c1", "Não quero mais falar sobre a licença do Serum 2.")
        turn = agent.before_response(
            "c1", "Por que eu escolhi o preset Serum 2?", ("Falamos do Serum 2.",)
        )
        payload = json.dumps(turn.as_dict(), ensure_ascii=False).lower()
        self.assertNotIn("licença", payload)
        self.assertIn("detalhe restrito", payload)
        self.assertIn("m2", turn.sensitivity.blocked_ids)


if __name__ == "__main__":
    unittest.main()
