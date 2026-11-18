"""Cliente mínimo da API DeepSeek (somente biblioteca padrão).

Identificador e limites verificados na documentação oficial em 2026-11-16:

- modelo ``deepseek-flash`` = DeepSeek-V4.1-Flash-1017;
- contexto de 1M tokens;
- saída máxima de 384K tokens (``max_tokens`` até 393216);
- endpoint OpenAI-compatible ``POST https://api.deepseek.com/chat/completions``;
- chave lida de ``DEEPSEEK_API_KEY``.

O modo de raciocínio vem ligado por padrão na API; aqui ele é desligado por
padrão para respostas de chat mais rápidas e baratas, e pode ser reativado
na configuração.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_MODEL_VERSION = "DeepSeek-V4.1-Flash-1017"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_CONTEXT_WINDOW = 1_000_000
MAX_OUTPUT_TOKENS = 393_216
DEFAULT_OUTPUT_RESERVE = 4_096
SUPPORTED_MODELS = ("deepseek-flash", "deepseek-v4-pro")
API_KEY_ENV = "DEEPSEEK_API_KEY"


class MissingApiKey(RuntimeError):
    """A variável DEEPSEEK_API_KEY não foi definida."""


class DeepSeekAPIError(RuntimeError):
    """Falha HTTP ou de rede ao chamar a API."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class DeepSeekSettings:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    context_window: int = DEFAULT_CONTEXT_WINDOW
    output_reserve: int = DEFAULT_OUTPUT_RESERVE
    max_tokens: int = DEFAULT_OUTPUT_RESERVE
    thinking: str = "disabled"
    reasoning_effort: str = "high"
    temperature: float = 0.7
    timeout: float = 60.0

    def validate(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo não suportado: {self.model!r}; use um de {SUPPORTED_MODELS}"
            )
        if self.context_window <= 0:
            raise ValueError("context_window deve ser positivo")
        if self.output_reserve <= 0:
            raise ValueError("output_reserve deve ser positivo")
        if self.output_reserve >= self.context_window:
            raise ValueError("output_reserve deve ser menor que context_window")
        if not 1 <= self.max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens deve estar entre 1 e {MAX_OUTPUT_TOKENS}")
        if self.max_tokens > self.output_reserve:
            raise ValueError("max_tokens não pode exceder output_reserve")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("thinking deve ser 'enabled' ou 'disabled'")
        if self.reasoning_effort not in {"none", "low", "high", "max"}:
            raise ValueError("reasoning_effort inválido")

    @property
    def model_version(self) -> str:
        return DEFAULT_MODEL_VERSION if self.model == DEFAULT_MODEL else "DeepSeek-V4-Pro-0813"


def load_deepseek_settings(config: Mapping[str, Any] | None = None) -> DeepSeekSettings:
    raw = dict((config or {}).get("deepseek", {}) or {})
    fields = set(DeepSeekSettings.__dataclass_fields__)
    values = {key: raw[key] for key in raw if key in fields}
    if "context_window" in values:
        values["context_window"] = int(values["context_window"])
    if "output_reserve" in values:
        values["output_reserve"] = int(values["output_reserve"])
    if "max_tokens" in values:
        values["max_tokens"] = int(values["max_tokens"])
    if "temperature" in values:
        values["temperature"] = float(values["temperature"])
    if "timeout" in values:
        values["timeout"] = float(values["timeout"])
    settings = DeepSeekSettings(**values)
    settings.validate()
    return settings


def resolve_api_key(explicit: str | None = None) -> str:
    key = (explicit or os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise MissingApiKey(
            f"Defina {API_KEY_ENV} com a chave da API DeepSeek "
            "(ou passe api_key explicitamente). Nada foi enviado."
        )
    return key


@dataclass(frozen=True)
class ChatCompletionResponse:
    content: str
    model: str
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "reasoning_content": self.reasoning_content,
            "usage": dict(self.usage),
        }


class ChatTransport(Protocol):
    name: str
    simulated: bool

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse: ...


def build_request_body(
    messages: Sequence[Mapping[str, str]],
    settings: DeepSeekSettings,
) -> dict[str, Any]:
    """Corpo exato enviado a ``/chat/completions``."""
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ],
        "max_tokens": settings.max_tokens,
        "stream": False,
    }
    if settings.thinking == "disabled":
        body["thinking"] = {"type": "disabled"}
        body["temperature"] = settings.temperature
    else:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = settings.reasoning_effort
    return body


def parse_completion(payload: Mapping[str, Any]) -> ChatCompletionResponse:
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    return ChatCompletionResponse(
        content=message.get("content") or "",
        model=str(payload.get("model", "")),
        finish_reason=choice.get("finish_reason"),
        reasoning_content=message.get("reasoning_content"),
        usage=dict(payload.get("usage") or {}),
        raw=dict(payload),
    )


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass
class DeepSeekClient:
    """Cliente HTTP real. ``opener`` permite testar sem rede."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    opener: Callable[[urllib.request.Request, float], bytes] = _urlopen
    name: str = "deepseek-http"
    simulated: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        opener: Callable[[urllib.request.Request, float], bytes] | None = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.opener = opener or _urlopen

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse:
        data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            raw = self.opener(request, timeout)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise DeepSeekAPIError(
                f"HTTP {error.code} da API DeepSeek: {body[:500]}",
                status=error.code,
                body=body,
            ) from error
        except urllib.error.URLError as error:
            raise DeepSeekAPIError(f"Falha de rede ao chamar a API DeepSeek: {error}") from error
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DeepSeekAPIError("Resposta da API não é JSON válido") from error
        return parse_completion(payload)


@dataclass
class SimulatedDeepSeekClient:
    """Transporte simulado: registra a requisição final e devolve uma resposta."""

    reply: str = "Resposta simulada do DeepSeek."
    reasoning: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    name: str = "simulated-deepseek-client"
    simulated: bool = True

    def complete(
        self, request_body: Mapping[str, Any], timeout: float = 60.0
    ) -> ChatCompletionResponse:
        # Cópia via JSON reproduz o que seria serializado no fio.
        recorded = json.loads(json.dumps(request_body, ensure_ascii=False))
        self.requests.append(recorded)
        return ChatCompletionResponse(
            content=self.reply,
            model=str(request_body.get("model", "")),
            finish_reason="stop",
            reasoning_content=self.reasoning or None,
            usage=dict(self.usage),
            raw={"simulated": True, "request": recorded},
        )
