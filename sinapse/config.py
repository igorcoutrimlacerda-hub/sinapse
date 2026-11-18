"""Configuração padrão e carregamento de JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .text import DIMENSIONS

DEFAULT_CONFIG: dict[str, Any] = {
    "base_weights": {
        "o_que": 0.25,
        "quem": 0.10,
        "onde": 0.05,
        "quando": 0.05,
        "como": 0.10,
        "por_que": 0.10,
        "para_que": 0.15,
    },
    "focus_weight": 1.0,
    "radius": 0.70,
    "top_k": 3,
    "attention_threshold": 0.05,
    "group_threshold": 0.33,
    "link_association_floor": 0.90,
    "window": 2048,
    "output_reserve": 512,
    "instructions": (
        "Você é um assistente com memória. Use o contexto de memória quando "
        "ele ajudar a responder ao turno atual."
    ),
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = {
        **DEFAULT_CONFIG,
        "base_weights": dict(DEFAULT_CONFIG["base_weights"]),
    }
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config.update(data)
        if "base_weights" in data:
            weights = dict(DEFAULT_CONFIG["base_weights"])
            weights.update(data["base_weights"])
            config["base_weights"] = weights
    for dimension in DIMENSIONS:
        config["base_weights"].setdefault(dimension, 0.0)
    return config
