"""Comparação local da antecipação ligada/desligada."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from .deepseek import ChatCompletionResponse
from .project_copilot import ProjectStore, chat_with_deepseek, local_answer


class _CaptureClient:
    simulated = False
    name = "anticipation-benchmark"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.used_ids: list[str] = []

    def complete(self, request_body, timeout: float = 60.0):
        self.requests.append(request_body)
        self.used_ids = [
            token
            for message in request_body.get("messages", ())
            for token in str(message.get("content", "")).split()
            if token.startswith("[m") or token.startswith("id=m")
        ]
        return ChatCompletionResponse(
            content="Resposta simulada.",
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
            raw={"simulated": True},
        )


def _seed(store: ProjectStore) -> None:
    for text in (
        "Comecei a Sinapse com o preset Pigments no bass.",
        "Falta testar a ponte em 96 kHz.",
        "Se a CPU ficar abaixo de 30%, o próximo passo é exportar; senão, reduzo o oversampling.",
        "As configurações de exportação são 48 kHz e 24 bits.",
    ):
        store.add_message(store.load(), text)


def _run_condition(enabled: bool) -> dict[str, Any]:
    config = load_config()
    config["anticipation"] = {
        "enabled": enabled,
        "max_rounds": 1,
        "max_candidates": 8,
        "max_time_ms": 150,
        "valid_turns": 1,
    }
    with tempfile.TemporaryDirectory() as directory:
        store = ProjectStore(directory)
        store.create("Sinapse")
        _seed(store)
        started = time.perf_counter()
        project = store.load()
        store.add_message(project, "A CPU ficou em 27%.")
        thread = None
        if enabled:
            thread = store.start_anticipation(
                "sinapse", "A CPU ficou em 27%.", config
            )
        current_ms = (time.perf_counter() - started) * 1000
        if thread is not None:
            thread.join(timeout=2)
        project = store.load()
        board = project.get("anticipatory") or {}
        reserved_ids = list(board.get("source_ids", ()))
        next_started = time.perf_counter()
        reservation = (
            store.consume_anticipation("sinapse", "Como exporto em 48 kHz?")
            if enabled
            else None
        )
        project = store.load()
        store.add_message(project, "Como exporto em 48 kHz?")
        project = store.load()
        client = _CaptureClient()
        chat_with_deepseek(
            project,
            "Como exporto em 48 kHz?",
            config,
            reserved_ids=reserved_ids,
            client=client,
            store=store,
        )
        next_ms = (time.perf_counter() - next_started) * 1000
        sent = "\n".join(
            str(message.get("content", ""))
            for message in (client.requests[0]["messages"] if client.requests else ())
        )
        return {
            "enabled": enabled,
            "current_ms": round(current_ms, 2),
            "next_ms": round(next_ms, 2),
            "anticipation_status": board.get("status"),
            "candidates_reserved": len(reserved_ids),
            "reserved_ids": reserved_ids,
            "memories_used": sum(1 for source in reserved_ids if source in sent),
            "evidence_coverage": int("configurações de exportação" in sent),
            "expired_predictions": int(
                board.get("status") == "expired" or board.get("expired_reason") is not None
            ),
            "anticipation_elapsed_ms": board.get("elapsed_ms"),
            "state_decision": project["state"]["decisions"][-1]["value"],
            "state_pending": project["state"]["pending"][0]["status"],
        }


def run_anticipation_benchmark() -> dict[str, Any]:
    return {
        "off": _run_condition(False),
        "on": _run_condition(True),
        "note": "Mesmos turnos e orçamento; cliente de chat simulado; DeepSeek não chamado.",
    }


def export_anticipation_benchmark(
    result: Mapping[str, Any],
    path: str | Path = "benchmark_out_anticipation/anticipation_benchmark.json",
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)
