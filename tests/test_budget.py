"""Testes do orçamento: contagem, encaixe, semente, IDs, dedup e diagnósticos."""

from __future__ import annotations

import unittest
from pathlib import Path

from sinapse.budget import (
    BaseInput,
    CharCounter,
    RecentTurn,
    SimulatedCounter,
    render_evidence_block,
    render_seed_block,
    select_context,
    wrap_memory,
)
from sinapse.config import load_config
from sinapse.memory import recall
from sinapse.models import Candidate, Message, load_messages
from sinapse.text import DIMENSIONS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo_conversation.json"
CONFIG = ROOT / "data" / "config.json"

POR_QUE = "Por que eu escolhi o Serum 2?"
COMO = "Como eu testei o bass?"
SERUM_CONTEXT = ["Estávamos falando do Serum 2 e do bass."]


def make_message(
    id: str,
    text: str,
    position: int = 1,
    reply_to: str | None = None,
    coordinates: dict | None = None,
) -> Message:
    coords = {d: tuple((coordinates or {}).get(d, ())) for d in DIMENSIONS}
    return Message(
        id=id,
        text=text,
        author="user",
        conversation_id="c",
        position=position,
        sent_at="2026-11-10T09:00:00-03:00",
        reply_to=reply_to,
        coordinates=coords,
    )


def make_candidate(
    message: Message,
    *,
    distance: float = 0.10,
    attention: float = 0.90,
    evidence: dict | None = None,
    within_radius: bool = True,
) -> Candidate:
    vector = {d: None for d in DIMENSIONS}
    vector["por_que"] = 1.0
    if evidence:
        vector.update(evidence)
    return Candidate(
        message=message,
        distance=distance,
        association=0.90,
        association_content=0.0,
        association_link=0.90,
        association_source="link",
        evidence=vector,
        attention=attention,
        within_radius=within_radius,
    )


def totals(base: BaseInput, blocks, counter) -> int:
    text = base.render() + "\n\n" + wrap_memory(blocks)
    return counter.count(text)


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.messages = load_messages(DATA)
        cls.config = load_config(CONFIG)
        cls.by_id = {m.id: m for m in cls.messages}

    def base(self, query: str, context: list[str] | None = None, *, ids=(), turns=None) -> BaseInput:
        if turns is None:
            turns = tuple(RecentTurn(text=t, author="user") for t in (context or []))
        return BaseInput(
            instructions=str(self.config["instructions"]),
            recent_turns=tuple(turns),
            current_turn=query,
            explicit_message_ids=tuple(ids),
        )

    def recall(self, query: str, context: list[str] | None = None):
        return recall(self.messages, query, context or [], self.config)

    def test_budget_formula_and_wide_selection(self) -> None:
        recall_result = self.recall(POR_QUE, SERUM_CONTEXT)
        base = self.base(POR_QUE, SERUM_CONTEXT)
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=2048,
            output_reserve=512,
            counter=SimulatedCounter(),
        )
        d = result.diagnostics
        self.assertEqual(d.memory_budget, d.capacity - d.base_tokens)
        self.assertEqual(d.selected_ids, ("m2",))
        self.assertTrue(d.seed_included)
        self.assertLessEqual(d.total_with_reserve, 2048)
        self.assertEqual(d.remaining_tokens, 2048 - 512 - d.used_tokens)
        self.assertEqual(d.status, "ok")

    def test_base_overflow_is_explicit(self) -> None:
        base = BaseInput(instructions="X" * 100, current_turn="Q")
        counter = CharCounter()
        base_tokens = counter.count(base.render())
        result = select_context(
            base,
            self.by_id["m1"],
            "por_que",
            [make_candidate(self.by_id["m2"])],
            window=10,
            output_reserve=5,
            counter=counter,
        )
        d = result.diagnostics
        self.assertEqual(d.status, "base_overflow")
        self.assertEqual(d.base_overflow_tokens, base_tokens - 5)
        self.assertTrue(d.exceeds_capacity)
        self.assertEqual(d.selected_ids, ())
        self.assertEqual(d.used_tokens, base_tokens)
        self.assertGreater(d.total_with_reserve, 10)
        self.assertEqual(result.context_text, base.render())
        self.assertTrue(all(dec.reason == "base_overflow" for dec in d.decisions))

    def test_zero_budget_selects_no_additional_memory(self) -> None:
        base = BaseInput(instructions="I", current_turn="Q")
        counter = CharCounter()
        base_tokens = counter.count(base.render())
        result = select_context(
            base,
            self.by_id["m1"],
            "por_que",
            [make_candidate(self.by_id["m2"])],
            window=base_tokens + 7,
            output_reserve=7,
            counter=counter,
        )
        d = result.diagnostics
        self.assertEqual(d.memory_budget, 0)
        self.assertEqual(d.status, "no_budget")
        self.assertEqual(d.selected_ids, ())
        self.assertEqual(d.memory_tokens, 0)
        self.assertEqual(result.context_text, base.render())
        self.assertEqual(d.decisions[0].reason, "no_budget")

    def test_exact_fit_and_one_token_over(self) -> None:
        seed = make_message("s", "alpha base", coordinates={"o_que": ["alpha"]})
        candidate_message = make_message(
            "c",
            "alpha motivo",
            position=2,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo"]},
        )
        candidate = make_candidate(candidate_message)
        base = BaseInput(instructions="I", current_turn="Q")
        counter = CharCounter()
        block = wrap_memory([render_evidence_block(candidate_message, 1)])
        exact_total = counter.count(base.render() + "\n\n" + block)
        exact = select_context(
            base,
            seed,
            "por_que",
            [candidate],
            window=exact_total + 5,
            output_reserve=5,
            counter=counter,
        )
        self.assertEqual(exact.diagnostics.selected_ids, ("c",))
        self.assertEqual(exact.diagnostics.total_with_reserve, exact_total + 5)
        self.assertEqual(exact.diagnostics.used_tokens, exact_total)

        too_tight = select_context(
            base,
            seed,
            "por_que",
            [candidate],
            window=exact_total + 4,
            output_reserve=5,
            counter=counter,
        )
        self.assertEqual(too_tight.diagnostics.selected_ids, ())
        decision = too_tight.diagnostics.decisions[0]
        self.assertEqual(decision.reason, "does_not_fit")
        self.assertIn("faltam 1 tokens", decision.detail or "")

    def test_labels_and_separators_are_counted(self) -> None:
        seed = make_message("s", "alpha base", coordinates={"o_que": ["alpha"]})
        candidate_message = make_message(
            "c",
            "alpha motivo",
            position=2,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo"]},
        )
        candidate = make_candidate(candidate_message)
        base = BaseInput(instructions="I", current_turn="Q")
        counter = CharCounter()
        base_tokens = counter.count(base.render())
        rendered = counter.count(render_evidence_block(candidate_message, 1))
        self.assertGreater(rendered, len(candidate_message.text))
        raw_capacity = base_tokens + len(candidate_message.text) + 2
        result = select_context(
            base,
            seed,
            "por_que",
            [candidate],
            window=raw_capacity + 5,
            output_reserve=5,
            counter=counter,
        )
        self.assertEqual(result.diagnostics.selected_ids, ())
        self.assertEqual(result.diagnostics.decisions[0].reason, "does_not_fit")

    def test_seed_optional_does_not_block_self_sufficient_evidence(self) -> None:
        recall_result = self.recall(COMO, SERUM_CONTEXT)
        base = self.base(COMO, SERUM_CONTEXT)
        counter = CharCounter()
        m3 = self.by_id["m3"]
        capacity = totals(base, [render_evidence_block(m3, 1)], counter)
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=capacity + 5,
            output_reserve=5,
            counter=counter,
        )
        d = result.diagnostics
        self.assertEqual(d.selected_ids, ("m3",))
        self.assertFalse(d.seed_included)
        self.assertEqual(d.seed_source, "omitted")
        self.assertIn("m3", result.context_text)

    def test_dependent_reply_requires_seed_pair(self) -> None:
        recall_result = self.recall(POR_QUE, SERUM_CONTEXT)
        base = self.base(POR_QUE, SERUM_CONTEXT)
        counter = CharCounter()
        m2 = self.by_id["m2"]
        evidence_only = totals(base, [render_evidence_block(m2, 1)], counter)
        tight = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=evidence_only + 5,
            output_reserve=5,
            counter=counter,
        )
        decision = next(d for d in tight.diagnostics.decisions if d.id == "m2")
        self.assertEqual(decision.reason, "does_not_fit")
        self.assertNotIn("m2", tight.diagnostics.selected_ids)

        pair_total = totals(
            base,
            [render_seed_block(recall_result.seed), render_evidence_block(m2, 1)],
            counter,
        )
        pair = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=pair_total + 5,
            output_reserve=5,
            counter=counter,
        )
        self.assertEqual(pair.diagnostics.selected_ids, ("m2",))
        self.assertTrue(pair.diagnostics.seed_included)
        self.assertEqual(pair.diagnostics.seed_source, "memory_required")

    def test_seed_in_base_is_reused_without_double_count(self) -> None:
        recall_result = self.recall(POR_QUE, SERUM_CONTEXT)
        m1 = self.by_id["m1"]
        base = BaseInput(
            instructions="I",
            recent_turns=(
                RecentTurn(text=m1.text, id="m1", author="user", position=1),
            ),
            current_turn=POR_QUE,
        )
        counter = CharCounter()
        m2 = self.by_id["m2"]
        capacity = totals(base, [render_evidence_block(m2, 1)], counter)
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=capacity + 5,
            output_reserve=5,
            counter=counter,
        )
        d = result.diagnostics
        self.assertEqual(d.selected_ids, ("m2",))
        self.assertTrue(d.seed_in_base)
        self.assertFalse(d.seed_included)
        self.assertEqual(d.seed_source, "base")
        self.assertEqual(d.seed_tokens, 0)
        self.assertIn("|m1|", result.context_text)
        self.assertIn("Escolhi o Serum 2 para fazer o bass.", result.context_text)

    def test_two_budgets_change_selection_not_attention(self) -> None:
        recall_result = self.recall(POR_QUE, SERUM_CONTEXT)
        base = self.base(POR_QUE, SERUM_CONTEXT)
        counter = SimulatedCounter()
        wide = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=2048,
            output_reserve=512,
            counter=counter,
        )
        base_tokens = counter.count(base.render())
        seed_only = counter.count(
            base.render() + "\n\n" + wrap_memory([render_seed_block(recall_result.seed)])
        ) - base_tokens
        tight = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=base_tokens + 512 + seed_only,
            output_reserve=512,
            counter=counter,
        )
        self.assertEqual(wide.diagnostics.selected_ids, ("m2",))
        self.assertEqual(tight.diagnostics.selected_ids, ())
        self.assertTrue(tight.diagnostics.seed_included)
        wide_attention = {c.message.id: c.attention for c in recall_result.candidates}
        tight_attention = {c.message.id: c.attention for c in recall_result.candidates}
        self.assertEqual(wide_attention, tight_attention)

    def test_long_candidate_skipped_in_favor_of_short(self) -> None:
        seed = make_message("s", "alpha base", coordinates={"o_que": ["alpha"]})
        long_message = make_message(
            "long",
            "alpha " + ("palavra " * 50),
            position=2,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo longo"]},
        )
        short_message = make_message(
            "short",
            "alpha curto",
            position=3,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo curto"]},
        )
        long_candidate = make_candidate(long_message, distance=0.10, attention=0.95)
        short_candidate = make_candidate(short_message, distance=0.20, attention=0.90)
        base = BaseInput(instructions="I", current_turn="Q")
        counter = CharCounter()
        capacity = totals(base, [render_evidence_block(short_message, 1)], counter)
        result = select_context(
            base,
            seed,
            "por_que",
            [long_candidate, short_candidate],
            window=capacity + 5,
            output_reserve=5,
            counter=counter,
        )
        decisions = {d.id: d for d in result.diagnostics.decisions}
        self.assertEqual(decisions["long"].reason, "does_not_fit")
        self.assertEqual(result.diagnostics.selected_ids, ("short",))

    def test_ids_already_in_base_are_skipped(self) -> None:
        recall_result = self.recall(COMO, SERUM_CONTEXT)
        m3 = self.by_id["m3"]
        base = BaseInput(
            instructions="I",
            recent_turns=(RecentTurn(text=m3.text, id="m3", author="user"),),
            current_turn=COMO,
        )
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=4000,
            output_reserve=512,
            counter=CharCounter(),
        )
        decision = next(d for d in result.diagnostics.decisions if d.id == "m3")
        self.assertEqual(decision.reason, "already_in_base")
        self.assertNotIn("m3", result.diagnostics.selected_ids)

    def test_explicit_base_message_id_is_recognized(self) -> None:
        recall_result = self.recall(COMO, SERUM_CONTEXT)
        base = BaseInput(
            instructions="I",
            current_turn=COMO,
            explicit_message_ids=("m3",),
        )
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=4000,
            output_reserve=512,
            counter=CharCounter(),
        )
        decision = next(d for d in result.diagnostics.decisions if d.id == "m3")
        self.assertEqual(decision.reason, "already_in_base")
        self.assertTrue(result.diagnostics.seed_in_base is False)

    def test_other_motive_does_not_replace_linked_motive(self) -> None:
        recall_result = recall(
            self.messages,
            POR_QUE,
            SERUM_CONTEXT,
            self.config,
            radius=0.95,
        )
        result = select_context(
            self.base(POR_QUE, SERUM_CONTEXT),
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=2048,
            output_reserve=512,
            counter=SimulatedCounter(),
        )
        self.assertEqual(result.diagnostics.selected_ids, ("m2",))
        m6 = next(d for d in result.diagnostics.decisions if d.id == "m6")
        self.assertTrue(m6.within_radius)
        self.assertEqual(m6.attention, 0.0)
        self.assertEqual(m6.reason, "below_threshold")
        self.assertNotIn("m6", result.diagnostics.selected_ids)

    def test_conservative_dedup_keeps_negation(self) -> None:
        seed = make_message("s", "Serum 2", coordinates={"o_que": ["Serum 2"]})
        first = make_message(
            "a", "Escolhi o Serum 2", position=2,
            coordinates={"o_que": ["Serum 2"]},
        )
        second = make_message(
            "b", "Não escolhi o Serum 2", position=3,
            coordinates={"o_que": ["Serum 2"]},
        )
        duplicate = make_message(
            "c", "Escolhi o Serum 2.", position=4,
            coordinates={"o_que": ["Serum 2"]},
        )
        candidates = [
            make_candidate(first, distance=0.10, attention=0.90),
            make_candidate(second, distance=0.11, attention=0.85),
            make_candidate(duplicate, distance=0.12, attention=0.80),
        ]
        base = BaseInput(instructions="I", current_turn="Q")
        result = select_context(
            base,
            seed,
            "por_que",
            candidates,
            window=10000,
            output_reserve=0,
            counter=CharCounter(),
            top_k=3,
        )
        decisions = {d.id: d for d in result.diagnostics.decisions}
        self.assertEqual(result.diagnostics.selected_ids, ("a", "b"))
        self.assertEqual(decisions["b"].reason, "selected")
        self.assertEqual(decisions["c"].reason, "duplicate")

    def test_top_k_and_deterministic_tie_break(self) -> None:
        seed = make_message("s", "alpha base", coordinates={"o_que": ["alpha"]})
        far = make_message(
            "far", "alpha longe", position=2,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo"]},
        )
        near = make_message(
            "near", "alpha perto", position=3,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo"]},
        )
        result = select_context(
            BaseInput(instructions="I", current_turn="Q"),
            seed,
            "por_que",
            [
                make_candidate(far, distance=0.20, attention=0.90),
                make_candidate(near, distance=0.10, attention=0.90),
            ],
            window=10000,
            output_reserve=0,
            counter=CharCounter(),
            top_k=1,
        )
        self.assertEqual(result.diagnostics.selected_ids, ("near",))
        decisions = {d.id: d for d in result.diagnostics.decisions}
        self.assertEqual(decisions["far"].reason, "top_k_full")

    def test_close_candidate_without_focus_evidence_stays_in_diagnosis(self) -> None:
        recall_result = self.recall(COMO, SERUM_CONTEXT)
        base = self.base(COMO, SERUM_CONTEXT)
        result = select_context(
            base,
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=2048,
            output_reserve=512,
            counter=SimulatedCounter(),
        )
        decision = next(d for d in result.diagnostics.decisions if d.id == "m2")
        self.assertTrue(decision.within_radius)
        self.assertEqual(decision.attention, 0.0)
        self.assertEqual(decision.reason, "no_focus_evidence")
        self.assertNotIn("m2", result.diagnostics.selected_ids)

    def test_diagnostics_are_not_in_delivered_context(self) -> None:
        recall_result = self.recall(POR_QUE, SERUM_CONTEXT)
        result = select_context(
            self.base(POR_QUE, SERUM_CONTEXT),
            recall_result.seed,
            recall_result.focus,
            recall_result.candidates,
            window=2048,
            output_reserve=512,
            counter=SimulatedCounter(),
        )
        self.assertIn("id=m2", result.context_text)
        self.assertIn("Porque eu já tinha comprado a licença.", result.context_text)
        for forbidden in ("dist=", "atenção", "motivo=", "does_not_fit", "vetor="):
            self.assertNotIn(forbidden, result.context_text)

    def test_counter_identity_is_reported_and_injectable(self) -> None:
        class ModelCounter:
            name = "fake-model-tokenizer"
            simulated = False

            def count(self, text: str) -> int:
                return len(text.split())

        seed = make_message("s", "alpha base", coordinates={"o_que": ["alpha"]})
        candidate_message = make_message(
            "c", "alpha motivo", position=2,
            coordinates={"o_que": ["alpha"], "por_que": ["motivo"]},
        )
        result = select_context(
            BaseInput(instructions="I", current_turn="Q"),
            seed,
            "por_que",
            [make_candidate(candidate_message)],
            window=1000,
            output_reserve=100,
            counter=ModelCounter(),
        )
        self.assertEqual(result.diagnostics.counter_name, "fake-model-tokenizer")
        self.assertFalse(result.diagnostics.counter_simulated)

        simulated = select_context(
            BaseInput(instructions="I", current_turn="Q"),
            seed,
            "por_que",
            [make_candidate(candidate_message)],
            window=1000,
            output_reserve=100,
            counter=SimulatedCounter(),
        )
        self.assertTrue(simulated.diagnostics.counter_simulated)
        self.assertEqual(simulated.diagnostics.counter_name, "simulated-words-punct-v1")


if __name__ == "__main__":
    unittest.main()
