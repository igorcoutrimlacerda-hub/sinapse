"""Configuração global do Farol para o DeepSeek.

A chave salva fica fora dos projetos e nunca é devolvida pela API da GUI.
``DEEPSEEK_API_KEY`` continua sendo a alternativa quando não há chave salva.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .config import load_config
from .deepseek import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_VERSION,
    DeepSeekClient,
    DeepSeekSettings,
    load_deepseek_settings,
)

DEFAULT_PATH = "data/farol_settings.json"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _redact(message: str, key: str | None) -> str:
    if key:
        message = message.replace(key, "[chave redigida]")
    return message[:500]


def redact_key(message: str, key: str | None) -> str:
    """Remove a chave de qualquer mensagem antes de ir para a GUI ou o log."""
    return _redact(str(message), key)


class FarolSettings:
    def __init__(
        self,
        path: str | Path = DEFAULT_PATH,
        *,
        config: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.config = dict(config or load_config())
        self.environ = environ if environ is not None else os.environ
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @property
    def model(self) -> str:
        return str(self.data.get("model") or DEFAULT_MODEL)

    def active_key(self) -> tuple[str | None, str]:
        saved = str(self.data.get("api_key") or "").strip()
        if saved:
            return saved, "panel"
        environment = str(self.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if environment:
            return environment, "environment"
        return None, "none"

    def status(self) -> dict[str, Any]:
        key, source = self.active_key()
        return {
            "configured": bool(key),
            "mode": "deepseek" if key else "local",
            "source": source,
            "model": self.model,
            "model_version": (
                DEFAULT_MODEL_VERSION if self.model == DEFAULT_MODEL else None
            ),
            "has_saved_key": bool(str(self.data.get("api_key") or "").strip()),
            "environment_available": bool(
                str(self.environ.get("DEEPSEEK_API_KEY") or "").strip()
            ),
        }

    def save_key(self, api_key: str, model: str | None = None) -> dict[str, Any]:
        key = str(api_key or "").strip()
        if not key:
            raise ValueError("Informe uma chave para salvar")
        self.data["api_key"] = key
        if model is not None and str(model).strip():
            self.data["model"] = str(model).strip()
        self.data["updated_at"] = _now()
        self._save()
        return self.status()

    def remove_key(self) -> dict[str, Any]:
        self.data.pop("api_key", None)
        self.data["updated_at"] = _now()
        self._save()
        return self.status()

    def set_model(self, model: str) -> dict[str, Any]:
        value = str(model or "").strip()
        if not value:
            raise ValueError("Informe um modelo")
        self.data["model"] = value
        self.data["updated_at"] = _now()
        self._save()
        return self.status()

    def test_connection(self, client=None) -> dict[str, Any]:
        key, source = self.active_key()
        if not key:
            return {
                "ok": False,
                "source": "none",
                "mode": "local",
                "error": "Nenhuma chave configurada. Salve uma chave ou defina DEEPSEEK_API_KEY.",
            }
        settings: DeepSeekSettings = load_deepseek_settings(self.config)
        settings = replace(settings, model=self.model)
        request = {
            "model": settings.model,
            "messages": [{"role": "user", "content": "Responda apenas: ok"}],
            "max_tokens": 8,
            "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
        }
        try:
            if client is None:
                client = DeepSeekClient(
                    api_key=key, base_url=settings.base_url
                )
            response = client.complete(request, timeout=settings.timeout)
            return {
                "ok": True,
                "source": source,
                "mode": "deepseek",
                "model": settings.model,
                "model_version": (
                    DEFAULT_MODEL_VERSION
                    if settings.model == DEFAULT_MODEL
                    else None
                ),
                "message": f"Conexão OK com {settings.model}.",
                "response_model": getattr(response, "model", None),
            }
        except Exception as error:
            return {
                "ok": False,
                "source": source,
                "mode": "deepseek",
                "model": settings.model,
                "error": _redact(str(error), key),
            }


def active_for_chat(
    store: FarolSettings, config: Mapping[str, Any]
) -> tuple[str | None, str, str]:
    key, source = store.active_key()
    return key, source, store.model
