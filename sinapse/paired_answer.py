"""Benchmark pareado: seletor atual vs seletor reduzido.

Mesma busca, ação de curiosidade congelada, candidatos, sensibilidade,
renderizador, orçamento, modelo e parâmetros. Só o conjunto de memória muda.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import SimulatedCounter
from .chat import EstimatedChatCounter
from .config import load_config
from .context_reduction import (
    _extra_eval_cases,
    _initial_state,
    _reduction,
    _run_action,
    _train_more_model,
)
from .curiosity_learning import ACTIONS, CuriosityCase, _load, _recent
from .curiosity_round2 import _expected_and_context
from .curiosity_round3 import _eval_cases as round3_eval_cases
from .deepseek import (
    ChatCompletionResponse,
    ChatTransport,
    DeepSeekClient,
    DeepSeekSettings,
    build_request_body,
    load_deepseek_settings,
)
from .text import normalize_text

BASE = "data/agent_conversation.json"


@dataclass(frozen=True)
class PairMeta:
    item_terms: tuple[str, ...]
    required_facts: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    forbidden_claims: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    ambiguous: bool
    reference_answer: str


META: dict[str, PairMeta] = {
    "ev_follow": PairMeta(("bicicleta",), ("trilha",), (), (), (), False, "Você comprou a bicicleta porque a trilha é mais curta (p2)."),
    "ev_follow_piano": PairMeta(("piano", "Piano V"), ("promoção",), (), (), (), False, "Você escolheu o Piano V porque estava em promoção (m6)."),
    "ev_correction": PairMeta(("Pigments",), ("Pigments",), (), (), (), False, "Agora você usa o Pigments (m10)."),
    "ev_price_link": PairMeta(("Serum 2",), ("280",), (), (), (), False, "O Serum 2 saiu por 280 reais (m20)."),
    "ev_many_links": PairMeta(("Pigments",), ("Pigments",), (), (), (), False, "Agora você usa o Pigments (m10)."),
    "ev_other_theme": PairMeta(("pack",), (), (), (), (), False, "Você preparou o pack do bass (m9)."),
    "ev_history": PairMeta(("Serum 2",), ("licença",), (), (), (), False, "Você escolheu o Serum 2 porque já tinha comprado a licença (m2)."),
    "ev_restricted": PairMeta(("Serum 2",), (), (), ("porque ele ja vinha", "porque ja vinha", "filtragem e o envelope configurados", "o motivo era", "escolhi porque"), ("licença", "licenca"), False, "Não tenho essa informação disponível."),
    "ev_self_sufficient": PairMeta(("Serum 2",), ("350",), (), (), (), False, "O Serum 2 custou 350 reais (m20)."),
    "ev_two_complementary": PairMeta(("Serum 2",), ("licença", "amigo"), (), (), (), False, "Você comprou a licença (m2) e um amigo indicou o Serum 2 (m10)."),
    "pack_quantity": PairMeta(("pack",), ("presets",), (("8", "oito"),), (), (), False, "O pack tem oito presets (m8)."),
    "pack_ambiguous": PairMeta(("pack",), (), (), (), (), True, "Você quer dizer a quantidade de presets (oito) ou como preparei o pack do bass?"),
}


def _pack_cases() -> list[CuriosityCase]:
    return [
        CuriosityCase(
            "pack_quantity",
            "pack_quantidade",
            _load(BASE),
            "c1",
            _recent("Quero saber quantos presets tem o pack."),
            "E aquilo do pack?",
            ("m8",),
            ("m7",),
            "m7",
            ("m8",),
            split="eval",
            memory_limit=120,
        ),
        CuriosityCase(
            "pack_ambiguous",
            "pack_ambiguo",
            _load(BASE),
            "c1",
            _recent("Falamos do pack."),
            "E aquilo do pack?",
            ("m8", "m9"),
            ("m7",),
            "m7",
            (),
            split="eval",
            memory_limit=160,
        ),
    ]


def _cases() -> list[CuriosityCase]:
    return list(round3_eval_cases()) + _extra_eval_cases() + _pack_cases()


def _answer_checks(meta: PairMeta, answer: str) -> dict[str, Any]:
    normalized = normalize_text(answer)
    facts = all(
        normalize_text(fact) in normalized for fact in meta.required_facts
    ) and all(
        any(normalize_text(option) in normalized for option in group)
        for group in meta.required_any
    )
    item = any(normalize_text(term) in normalized for term in meta.item_terms)
    unsupported = any(
        normalize_text(claim) in normalized for claim in meta.forbidden_claims
    )
    sensitive = any(
        normalize_text(term) in normalized for term in meta.forbidden_terms
    )
    ambiguity = True
    if meta.ambiguous:
        clarification = any(
            marker in normalized
            for marker in (
                "qual",
                "voce quer dizer",
                "quer dizer",
                "esclarec",
                "quer que eu",
                "quer retomar",
                "voce prefere",
                "ou revisar",
                "quer continuar",
                "continuar de onde",
                "ou detalhar",
                "prefere que eu",
                "quer revisar",
                "revisar algo",
                "algo especifico",
            )
        )
        interpretation = any(
            marker in normalized
            for marker in ("entendi", "interpret", "se for", "voce esta falando")
        )
        evidence = any(
            term in normalized
            for term in ("8", "oito", "presets", "bass", "divulgar", "preparei")
        )
        ambiguity = (clarification or interpretation) and evidence
    components = [
        1.0 if facts else 0.0,
        1.0 if item else 0.0,
        0.0 if unsupported else 1.0,
        0.0 if sensitive else 1.0,
        1.0 if ambiguity else 0.0,
    ]
    return {
        "facts_ok": facts,
        "item_ok": item,
        "unsupported": unsupported,
        "sensitive": sensitive,
        "ambiguity_ok": ambiguity,
        "score": round(sum(components) / len(components), 4),
    }


class SimulatedPairClient:
    name = "simulated-pair"
    simulated = True

    def __init__(self, meta: PairMeta, messages: Sequence[Mapping[str, str]]) -> None:
        self.meta = meta
        self.messages = messages
        self.requests: list[dict] = []

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse:
        self.requests.append(json.loads(json.dumps(request_body, ensure_ascii=False)))
        context = "\n".join(str(message["content"]) for message in self.messages[:-1])
        normalized = normalize_text(context)
        evidence = any(
            normalize_text(term) in normalized for term in self.meta.item_terms
        )
        content = self.meta.reference_answer if evidence else "Não tenho informação suficiente."
        return ChatCompletionResponse(
            content=content,
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={},
            raw={"simulated": True},
        )


def run_paired(
    config: Mapping[str, Any] | None = None,
    settings: DeepSeekSettings | None = None,
    client: ChatTransport | None = None,
) -> dict[str, Any]:
    config = dict(config or load_config())
    settings = settings or load_deepseek_settings(config)
    model = _train_more_model(config)
    counter = EstimatedChatCounter(SimulatedCounter())
    entries = []
    total_calls = 0
    for case in _cases():
        meta = META[case.id]
        state = _initial_state(case, config)
        action = max(
            ACTIONS, key=lambda item: _score_action(model, state, item)
        )
        current = _run_action(action, state, config)
        reduced = _reduction(state, current, config)
        pair = {}
        for label, messages in (("current", current["messages"]), ("reduced", reduced["messages"])):
            total_calls += 1
            request_body = build_request_body(messages, settings)
            if client is None:
                answer_client: ChatTransport = SimulatedPairClient(meta, messages)
            else:
                answer_client = client
            try:
                response = answer_client.complete(request_body, timeout=settings.timeout)
                answer = response.content
                usage = dict(response.usage)
                error = None
            except Exception as exc:
                answer = ""
                usage = {}
                error = str(exc)
            checks = _answer_checks(meta, answer)
            pair[label] = {
                "messages": [dict(message) for message in messages],
                "selected": list(
                    current["selected"] if label == "current" else reduced["selected"]
                ),
                "removed": [] if label == "current" else list(reduced["removed"]),
                "answer": answer,
                "checks": checks,
                "score": checks["score"],
                "token_source": "api" if usage.get("prompt_tokens") is not None else "estimativa_local",
                "request_tokens_estimate": counter.count_messages(messages),
                "memory_tokens": current["memory_tokens"] if label == "current" else reduced["memory_tokens"],
                "usage": usage,
                "error": error,
            }
        verdict = (
            "melhorou"
            if pair["reduced"]["score"] > pair["current"]["score"]
            else "piorou"
            if pair["reduced"]["score"] < pair["current"]["score"]
            else "empatou"
        )
        entries.append(
            {
                "case": case.id,
                "family": case.family,
                "action": action,
                "current": pair["current"],
                "reduced": pair["reduced"],
                "verdict": verdict,
                "removed": pair["reduced"]["removed"],
            }
        )
    counts = {
        verdict: sum(1 for entry in entries if entry["verdict"] == verdict)
        for verdict in ("melhorou", "empatou", "piorou")
    }
    return {
        "cases": len(entries),
        "calls": total_calls,
        "conditions": {
            "current": _aggregate(entries, "current"),
            "reduced": _aggregate(entries, "reduced"),
        },
        "counts": counts,
        "entries": entries,
    }


def _score_action(model, state, action: str) -> float:
    from .curiosity_round2 import probabilities
    from .curiosity_learning import _features

    return probabilities(model, _features(state))[action]


def _aggregate(entries: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    items = [entry[key] for entry in entries]
    return {
        "facts": sum(1 for item in items if item["checks"]["facts_ok"]),
        "item": sum(1 for item in items if item["checks"]["item_ok"]),
        "unsupported": sum(1 for item in items if item["checks"]["unsupported"]),
        "sensitive": sum(1 for item in items if item["checks"]["sensitive"]),
        "ambiguity": sum(1 for item in items if item["checks"]["ambiguity_ok"]),
        "avg_score": round(sum(item["score"] for item in items) / len(items), 4),
        "avg_memory_tokens": round(
            sum(item["memory_tokens"] for item in items) / len(items), 1
        ),
        "avg_request_tokens": round(
            sum(item["request_tokens_estimate"] for item in items) / len(items), 1
        ),
        "api_prompt_tokens": sum(
            int(item["usage"].get("prompt_tokens", 0) or 0) for item in items
        ),
        "cases": len(items),
    }


def export_paired(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    review = []
    for index, entry in enumerate(result["entries"], start=1):
        flipped = index % 2 == 0
        first, second = ("reduced", "current") if flipped else ("current", "reduced")
        review.append(
            {
                "review_id": f"P{index:02d}",
                "case": entry["case"],
                "answer_A": entry[first]["answer"],
                "answer_B": entry[second]["answer"],
                "rubric": "Compare fatos, atribuição ao item, suporte, sensibilidade e ambiguidade; não sabe qual condição é A ou B.",
            }
        )
    (out / "paired_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "results": str(out / "paired_results.json"),
        "review": str(out / "paired_review.json"),
    }
