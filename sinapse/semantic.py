"""Índice semântico local, determinístico e sem rede.

Representações numéricas por hashing de palavras, radicais, conceitos de
software e trigramas. É uma aproximação local para comparar significado quando
o vocabulário da pergunta difere do registro; nomes exatos continuam na busca
lexical.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from .text import normalize_text, tokenize

DIMENSIONS = 384

CONCEPTS: dict[str, tuple[str, ...]] = {
    "login": ("auth",),
    "autenticacao": ("auth",),
    "autenticar": ("auth",),
    "sessao": ("auth",),
    "sessoes": ("auth",),
    "token": ("auth",),
    "tokens": ("auth",),
    "jwt": ("auth",),
    "cookie": ("auth", "client"),
    "cookies": ("auth", "client"),
    "credencial": ("auth",),
    "chave": ("auth",),
    "chaves": ("auth",),
    "oauth": ("auth",),
    "csrf": ("auth",),
    "navegador": ("client",),
    "cliente": ("client",),
    "browser": ("client",),
    "frontend": ("client",),
    "front": ("client",),
    "servidor": ("server",),
    "server": ("server",),
    "desenvolvimento": ("development",),
    "development": ("development",),
    "aplicacao": ("app",),
    "aplicação": ("app",),
    "application": ("app",),
    "app": ("app",),
    "inicia": ("start",),
    "iniciar": ("start",),
    "start": ("start",),
    "run": ("start",),
    "loop": ("loop",),
    "evento": ("event",),
    "eventos": ("event",),
    "event": ("event",),
    "janela": ("window",),
    "window": ("window",),
    "python": ("python",),
    "interpretador": ("python",),
    "inicializado": ("start",),
    "inicializa": ("start",),
    "executa": ("start",),
    "executar": ("start",),
    "backend": ("server",),
    "api": ("server",),
    "revogar": ("revoke",),
    "revogacao": ("revoke",),
    "remover": ("revoke",),
    "removemos": ("revoke",),
    "abandonar": ("revoke",),
    "abandonamos": ("revoke",),
    "tirar": ("revoke",),
    "tiramos": ("revoke",),
    "invalidar": ("revoke",),
    "expirar": ("revoke",),
    "logout": ("revoke",),
    "contexto": ("context",),
    "fala": ("context",),
    "repetir": ("dedup",),
    "repeticao": ("dedup",),
    "duplicacao": ("dedup",),
    "duplicacoes": ("dedup",),
    "cache": ("cache",),
    "redis": ("cache",),
    "memoria": ("cache",),
    "fila": ("queue",),
    "evento": ("queue",),
    "eventos": ("queue",),
    "mensageria": ("queue",),
    "teste": ("test",),
    "testes": ("test",),
    "pytest": ("test",),
    "jest": ("test",),
    "suite": ("test",),
    "bug": ("bug",),
    "erro": ("bug",),
    "falha": ("bug",),
    "crash": ("bug",),
    "traceback": ("bug",),
    "deploy": ("release",),
    "release": ("release",),
    "producao": ("release",),
}


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def concept_terms(text: str) -> set[str]:
    terms = set(tokenize(text))
    for token in list(terms):
        for concept in CONCEPTS.get(token, ()):
            terms.add(concept)
    return terms


def _features(text: str) -> Iterable[tuple[str, float]]:
    normalized = normalize_text(text)
    for token in tokenize(text):
        stem = _stem(token)
        yield f"w:{stem}", 1.0
        for concept in CONCEPTS.get(token, ()) or CONCEPTS.get(stem, ()):
            yield f"c:{concept}", 1.7
        if len(token) >= 5:
            padded = f"^{token}$"
            for index in range(len(padded) - 2):
                yield f"g:{padded[index:index + 3]}", 0.35
    # trigramas de nomes técnicos também ajudam com plural/acento
    for token in normalized.split():
        if len(token) >= 6:
            for index in range(len(token) - 2):
                yield f"g:{token[index:index + 3]}", 0.15


def embed(text: str, dimensions: int = DIMENSIONS) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for feature, weight in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _entry_text(entry: Mapping[str, Any]) -> str:
    parts = [
        str(entry.get("value") or entry.get("text") or entry.get("to") or ""),
        str(entry.get("reason") or ""),
        " ".join(entry.get("files", ()) or ()),
        " ".join(entry.get("keywords", ()) or ()),
        str(entry.get("symptom") or ""),
        str(entry.get("command") or ""),
    ]
    return " ".join(part for part in parts if part)


class SemanticIndex:
    """Candidatos por significado; a validação continua no seletor."""

    def __init__(self, project: Mapping[str, Any]) -> None:
        self.entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in ("decisions", "bugs", "completed", "pending", "changes", "hypotheses"):
            for item in project.get("state", {}).get(key, ()):
                source_ids = [str(source) for source in item.get("source_ids", ())]
                marker = source_ids[0] if source_ids else str(item.get("entry_id") or "")
                if not marker or marker in seen:
                    continue
                seen.add(marker)
                self.entries.append(
                    {
                        "text": _entry_text(item),
                        "source_ids": source_ids,
                        "kind": item.get("kind") or key,
                        "entry_id": item.get("entry_id"),
                        "status": item.get("status"),
                        "raw": item,
                    }
                )
        for message in project.get("messages", ()):
            if message.get("author") == "assistant" and not message.get("kind"):
                continue
            source_id = str(message.get("id", ""))
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            self.entries.append(
                {
                    "text": _entry_text(message),
                    "source_ids": [source_id],
                    "kind": message.get("kind") or "note",
                    "entry_id": message.get("entry_id"),
                    "status": message.get("status"),
                    "raw": message,
                }
            )
        self.vectors = [embed(entry["text"]) for entry in self.entries]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.22,
        exclude: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = embed(query)
        excluded = exclude or set()
        scored = []
        for entry, vector in zip(self.entries, self.vectors):
            if excluded.intersection(entry["source_ids"]):
                continue
            score = cosine(query_vector, vector)
            if score >= min_score:
                scored.append({**entry, "score": round(score, 4)})
        scored.sort(key=lambda item: (-item["score"], item["source_ids"]))
        return scored[:limit]
