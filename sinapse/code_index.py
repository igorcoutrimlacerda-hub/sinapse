"""Importação e busca de código-fonte local associado ao projeto.

O índice é gravado ao lado dos projetos (`<id>.code.json`), com commit,
caminho, linhas, símbolo e licença. A leitura é somente de dados: nenhum
script do repositório é executado.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .redaction import contains_credential, redact_text
from .semantic import concept_terms, embed, cosine
from .text import normalize_text, tokenize

MAX_FILE_BYTES = 256 * 1024
MAX_LINE_BLOCK = 80
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "target",
    ".venv", "venv", "env", "__pycache__", ".mypy_cache", ".pytest_cache",
    "coverage", ".next", ".nuxt", "out", "bin", "obj",
}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".7z", ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".wasm",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".sqlite",
}
SECRET_NAMES = (".env", "id_rsa", "id_ed25519", "secrets", "credentials")
CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".scala", ".sh", ".ps1", ".sql",
}
DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}
TEST_HINTS = ("test", "spec", "tests", "__tests__", "fixtures")
BEHAVIOR_CUES = (
    "onde", "implementa", "implementado", "comportamento", "inicia", "iniciar",
    "executa", "executar", "servidor", "desenvolvimento", "função", "funcao",
    "classe", "método", "metodo", "fluxo", "retorna", "valida", "invalida",
    "sessão", "sessao", "token", "senha", "login", "aplicação", "aplicacao",
    "start", "run", "server", "development", "behavior", "function", "class",
    "method", "returns", "executes",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git falhou: {message[:300]}")
    return result.stdout


def repo_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    output = _run_git(candidate, "rev-parse", "--show-toplevel").decode(
        "utf-8", errors="replace"
    ).strip()
    return Path(output)


def resolve_commit(repo: Path, commit: str | None = None) -> str:
    reference = (commit or "HEAD").strip() or "HEAD"
    return _run_git(repo, "rev-parse", reference).decode("utf-8").strip()


def _tracked_files(repo: Path, commit: str) -> list[str]:
    output = _run_git(repo, "ls-tree", "-r", "-z", "--name-only", commit)
    return [item for item in output.decode("utf-8", errors="replace").split("\0") if item]


def _file_bytes(repo: Path, commit: str, path: str) -> bytes:
    return _run_git(repo, "show", f"{commit}:{path}")


def _is_excluded(path: str, data: bytes) -> bool:
    parts = Path(path).parts
    if any(part.lower() in EXCLUDED_DIRS for part in parts):
        return True
    lower = path.lower()
    if any(lower.endswith(suffix) for suffix in BINARY_SUFFIXES):
        return True
    if any(name in lower for name in SECRET_NAMES):
        return True
    if b"\x00" in data[:4096]:
        return True
    if len(data) > MAX_FILE_BYTES:
        return True
    return False


def _worktree_files(repo: Path) -> list[str]:
    output = _run_git(repo, "ls-files", "-z")
    return [
        path
        for path in output.decode("utf-8", errors="replace").split("\0")
        if path and (repo / path).is_file()
    ]


def _blob_reader(repo: Path, commit: str, paths: Sequence[str]):
    """Lê blobs em um único `git cat-file --batch` (escala)."""
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for path in paths:
            local = repo / path
            if local.is_file():
                try:
                    yield path, local.read_bytes()
                except OSError:
                    pass
                continue
            process.stdin.write(f"{commit}:{path}\n".encode("utf-8"))
            process.stdin.flush()
            header = process.stdout.readline()
            if not header or header.rstrip().endswith(b" missing"):
                continue
            parts = header.split()
            if len(parts) < 3:
                continue
            try:
                size = int(parts[2])
            except ValueError:
                continue
            if size > MAX_FILE_BYTES:
                remaining = size
                while remaining > 0:
                    chunk = process.stdout.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                process.stdout.read(1)
                yield path, None
                continue
            data = process.stdout.read(size)
            process.stdout.read(1)
            yield path, data
    finally:
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.stdout.close()
        except OSError:
            pass
        process.wait(timeout=10)


def _detect_license(repo: Path, commit: str, files: Sequence[str]) -> dict[str, Any] | None:
    candidates = [
        path for path in files
        if Path(path).name.upper().startswith(("LICENSE", "LICENCE", "COPYING"))
    ]
    for path in candidates[:2]:
        try:
            text = _file_bytes(repo, commit, path).decode("utf-8-sig", errors="replace")
        except ValueError:
            continue
        head = "\n".join(text.splitlines()[:40])
        patterns = (
            ("Apache-2.0", "Apache License"),
            ("MIT", "MIT License"),
            ("BSD-3-Clause", "BSD 3-Clause"),
            ("BSD-3-Clause", "Redistribution and use in source and binary forms"),
            ("BSD-2-Clause", "BSD 2-Clause"),
            ("GPL-3.0", "GNU GENERAL PUBLIC LICENSE"),
            ("GPL-2.0", "GNU GENERAL PUBLIC LICENSE"),
            ("MPL-2.0", "Mozilla Public License"),
            ("ISC", "ISC License"),
        )
        for name, marker in patterns:
            if marker.lower() in head.lower():
                return {"name": name, "path": path, "commit": commit}
        return {"name": "desconhecida", "path": path, "commit": commit}
    return None


def _language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
        ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
        ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".php": "php", ".swift": "swift", ".scala": "scala", ".sh": "shell",
        ".ps1": "powershell", ".sql": "sql",
    }.get(suffix, "text")


def _kind(path: str) -> str:
    lower = path.lower()
    if any(hint in lower for hint in TEST_HINTS):
        return "test"
    if Path(path).suffix.lower() in DOC_SUFFIXES:
        return "doc"
    if Path(path).suffix.lower() in CODE_SUFFIXES:
        return "code"
    return "text"


def _chunk_id(path: str, start: int, end: int, symbol: str | None) -> str:
    digest = hashlib.sha1(
        f"{path}|{start}|{end}|{symbol or ''}".encode("utf-8")
    ).hexdigest()[:12]
    return f"c-{digest}"


def _line_chunks(text: str, path: str, start: int = 1, end: int | None = None):
    lines = text.splitlines()
    end = end or len(lines)
    for offset in range(start - 1, end, MAX_LINE_BLOCK):
        block = lines[offset:offset + MAX_LINE_BLOCK]
        if not block:
            continue
        first = offset + 1
        last = offset + len(block)
        yield first, last, "\n".join(block)


def _python_chunks(text: str, path: str):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        yield from ((first, last, body, None) for first, last, body in _line_chunks(text, path))
        return
    lines = text.splitlines()
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        symbol = node.name
        body = "\n".join(lines[start - 1:end])
        yield start, end, body, symbol
        covered.update(range(start, end + 1))
    gaps = [index for index in range(1, len(lines) + 1) if index not in covered]
    for first, last, body in _line_chunks(text, path):
        if any(index in gaps for index in range(first, last + 1)):
            yield first, last, body, None


def _brace_chunks(text: str, path: str):
    lines = text.splitlines()
    pattern = re.compile(
        r"(?P<kind>function|class|def|func|fn|interface|struct|enum)\s+"
        r"(?P<name>[A-Za-z_$][\w$]*)"
        r"|(?:const|let|var)\s+(?P<arrow>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("
    )
    found = []
    for index, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if not match:
            continue
        name = match.group("name") or match.group("arrow")
        depth = 0
        end = index
        for cursor in range(index - 1, len(lines)):
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            end = cursor + 1
            if depth <= 0 and cursor > index - 1:
                break
            if depth <= 0 and "{" not in lines[cursor]:
                # declaração de uma linha
                end = cursor + 1
                break
        found.append((index, end, name))
    if not found:
        yield from ((first, last, body, None) for first, last, body in _line_chunks(text, path))
        return
    covered = set()
    for start, end, name in found:
        body = "\n".join(lines[start - 1:end])
        yield start, end, body, name
        covered.update(range(start, end + 1))
    for first, last, body in _line_chunks(text, path):
        if any(index not in covered for index in range(first, last + 1)):
            yield first, last, body, None


def _chunks_for(path: str, text: str) -> Iterable[tuple[int, int, str, str | None]]:
    suffix = Path(path).suffix.lower()
    kind = _kind(path)
    if kind in {"doc", "test"}:
        for first, last, body in _line_chunks(text, path):
            yield first, last, body, None
        return
    if suffix == ".py":
        for first, last, body, symbol in _python_chunks(text, path):
            yield first, last, body, symbol
        return
    if suffix in CODE_SUFFIXES:
        for first, last, body, symbol in _brace_chunks(text, path):
            yield first, last, body, symbol
        return
    for first, last, body in _line_chunks(text, path):
        yield first, last, body, None


def _comment_span(
    source: list[str], path: str, start: int, end: int, code_chunks: list[dict]
) -> str | None:
    local = [chunk for chunk in code_chunks if chunk.get("path") == path]
    for chunk in local:
        if (
            chunk.get("symbol")
            and chunk["start"] <= start
            and end <= chunk["end"]
        ):
            return chunk.get("symbol")
    for chunk in local:
        if chunk.get("symbol") and end < chunk["start"] and chunk["start"] - end <= 2:
            return chunk.get("symbol")
    for chunk in local:
        if chunk["start"] <= start and end <= chunk["end"]:
            return chunk.get("symbol")
    return None


def _comment_chunks(
    path: str, text: str, code_chunks: list[dict]
) -> list[dict]:
    """Comentários e docstrings como fontes próprias, sem executar nada."""
    source = text.splitlines()
    comments: list[tuple[int, int, str]] = []
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        import ast
        import io
        import tokenize

        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT and token.string.strip("# ").strip():
                    comments.append(
                        (token.start[0], token.end[0], token.string.strip("# ").strip())
                    )
        except (tokenize.TokenError, IndentationError):
            pass
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(node, "body", ())
                    if body and isinstance(body[0], ast.Expr) and isinstance(
                        getattr(body[0], "value", None), ast.Constant
                    ) and isinstance(body[0].value.value, str):
                        value = body[0].value
                        comments.append(
                            (
                                getattr(value, "lineno", 1),
                                getattr(value, "end_lineno", getattr(value, "lineno", 1)),
                                value.value.strip(),
                            )
                        )
        except SyntaxError:
            pass
    elif suffix in CODE_SUFFIXES:
        block = False
        start = 0
        buffer: list[str] = []
        for number, line in enumerate(source, start=1):
            stripped = line.strip()
            if "//" in line and not stripped.startswith(("http://", "https://")):
                value = line.split("//", 1)[1].strip()
                if value:
                    comments.append((number, number, value))
            if "/*" in line:
                block = True
                start = number
                buffer = [line.split("/*", 1)[1]]
            elif block:
                buffer.append(line)
            if block and "*/" in line:
                value = "\n".join(buffer).split("*/", 1)[0].strip()
                if value:
                    comments.append((start, number, value))
                block = False
                buffer = []
    chunks = []
    for start, end, value in comments:
        value = value.strip()
        if not value or contains_credential(value):
            continue
        symbol = _comment_span(source, path, start, end, code_chunks)
        chunks.append(
            {
                "id": _chunk_id(path, start, end, f"comment:{symbol or ''}"),
                "path": path,
                "start": start,
                "end": end,
                "symbol": symbol,
                "language": _language(path),
                "kind": "comment",
                "commit": "",
                "text": redact_text(value),
            }
        )
    return chunks


def _divergence(comment: str, code: str) -> str | None:
    """Sinaliza contradições simples entre comentário e implementação."""
    c = normalize_text(comment)
    body = normalize_text(code)
    upper_claim = any(word in c for word in ("maiuscul", "upper", "uppercase"))
    lower_claim = any(word in c for word in ("minuscul", "lower", "lowercase"))
    code_upper = bool(re.search(r"\.upper\(|\bupper\b", body))
    code_lower = bool(re.search(r"\.lower\(|\blower\b", body))
    if upper_claim and not lower_claim and code_lower and not code_upper:
        return "comentário diz maiúsculas; implementação usa minúsculas"
    if lower_claim and not upper_claim and code_upper and not code_lower:
        return "comentário diz minúsculas; implementação usa maiúsculas"
    if any(marker in c for marker in ("todo", "fixme", "obsoleto", "deprecated")):
        return "comentário marcado como pendente/obsoleto"
    return None


def index_path(root: str | Path, project_id: str) -> Path:
    return Path(root) / f"{project_id}.code.json"


def import_repository(
    store,
    project_id: str,
    *,
    repo_path: str | Path | None = None,
    commit: str | None = None,
) -> dict[str, Any]:
    """Indexa um commit Git e grava o índice ao lado do projeto."""
    project = store.load(project_id)
    target = Path(repo_path or project.get("repo_path") or "").expanduser()
    if not str(target):
        raise ValueError("Associe um repositório ao projeto antes de importar")
    repo = repo_root(target)
    resolved = resolve_commit(repo, commit)
    head_commit = resolve_commit(repo, "HEAD")
    if resolved == head_commit:
        files = _worktree_files(repo)
    else:
        files = _tracked_files(repo, resolved)
    license_info = _detect_license(repo, resolved, files)
    chunks = []
    indexed_files = []
    skipped = []
    for path, data in _blob_reader(repo, resolved, files):
        if data is None:
            skipped.append(path)
            continue
        if _is_excluded(path, data):
            skipped.append(path)
            continue
        text = data.decode("utf-8-sig", errors="replace")
        if contains_credential(text):
            skipped.append(path)
            continue
        language = _language(path)
        kind = _kind(path)
        indexed_files.append(path)
        file_chunks = []
        for start, end, body, symbol in _chunks_for(path, text):
            if not body.strip():
                continue
            file_chunks.append(
                {
                    "id": _chunk_id(path, start, end, symbol),
                    "path": path,
                    "start": start,
                    "end": end,
                    "symbol": symbol,
                    "language": language,
                    "kind": kind,
                    "commit": resolved,
                    "text": redact_text(body),
                }
            )
        for comment in _comment_chunks(path, text, file_chunks):
            comment["commit"] = resolved
            file_chunks.append(comment)
        chunks.extend(file_chunks)
    index = {
        "project_id": project_id,
        "repo_path": str(repo),
        "commit": resolved,
        "commit_short": resolved[:12],
        "imported_at": _now(),
        "license": license_info,
        "files": indexed_files,
        "skipped": skipped[:200],
        "chunks": chunks,
    }
    path = index_path(store.root, project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    project["code_index"] = {
        "repo_path": str(repo),
        "commit": resolved,
        "commit_short": resolved[:12],
        "imported_at": index["imported_at"],
        "license": license_info,
        "files": len(indexed_files),
        "chunks": len(chunks),
        "index_path": str(path),
    }
    store.save(project)
    store.diagnostics.record(
        project_id,
        "code.imported",
        details={
            "commit": resolved,
            "files": len(indexed_files),
            "chunks": len(chunks),
            "license": (license_info or {}).get("name"),
        },
    )
    return project["code_index"]


@dataclass
class CodeHit:
    chunk: Mapping[str, Any]
    score: float
    relation: str = "match"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk["id"],
            "path": self.chunk["path"],
            "start": self.chunk["start"],
            "end": self.chunk["end"],
            "symbol": self.chunk.get("symbol"),
            "kind": self.chunk.get("kind"),
            "commit": self.chunk.get("commit"),
            "score": round(self.score, 4),
            "relation": self.relation,
            "text": self.chunk.get("text", ""),
        }


class CodeIndex:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)
        self.chunks = list(data.get("chunks", ()))
        self.commit = str(data.get("commit", ""))
        self.repo_path = str(data.get("repo_path", ""))
        self.license = data.get("license")
        self._vectors: list[tuple[float, ...]] | None = None
        self._full_blobs: list[str] | None = None
        self._body_blobs: list[str] | None = None

    @classmethod
    def load(cls, project: Mapping[str, Any]) -> "CodeIndex | None":
        meta = project.get("code_index")
        if not meta or not meta.get("index_path"):
            return None
        path = Path(meta["index_path"])
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _blobs(self, body_only: bool = False) -> list[str]:
        if body_only:
            if self._body_blobs is None:
                self._body_blobs = [
                    (
                        str(chunk.get("path", ""))
                        + " "
                        + str(chunk.get("symbol") or "")
                        + " "
                        + str(chunk.get("text", ""))
                    ).lower()
                    for chunk in self.chunks
                ]
            return self._body_blobs
        if self._full_blobs is None:
            self._full_blobs = [
                (
                    str(chunk.get("path", ""))
                    + " "
                    + str(chunk.get("symbol") or "")
                    + " "
                    + str(chunk.get("text", ""))
                ).lower()
                for chunk in self.chunks
            ]
        return self._full_blobs

    def _embedding(self, index: int) -> tuple[float, ...]:
        if self._vectors is None:
            self._vectors = [
                embed(
                    " ".join(
                        part for part in (
                            str(chunk.get("symbol") or ""),
                            chunk.get("path", ""),
                            str(chunk.get("text", ""))[:900],
                        ) if part
                    )
                )
                for chunk in self.chunks
            ]
        return self._vectors[index]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        path_hint: str | None = None,
        symbol: str | None = None,
    ) -> list[CodeHit]:
        normalized = normalize_text(query)
        query_tokens = set(tokenize(query))
        identifiers = set(re.findall(r"[A-Za-z_$][\w$]{2,}", query))
        explicit_symbol = symbol or (next(iter(identifiers), None) if "`" in query else None)
        query_vector = embed(query)
        asks_callers = any(
            cue in normalized
            for cue in ("quem chama", "chama", "chamado", "usado", "usa", "referencia", "referência")
        )
        behavior_query = any(cue in normalized for cue in BEHAVIOR_CUES)
        exact_symbols = {
            str(chunk.get("symbol"))
            for chunk in self.chunks
            if chunk.get("symbol") and str(chunk.get("symbol")) in identifiers
        }
        if exact_symbols and not asks_callers:
            candidates = [
                CodeHit(chunk=chunk, score=5.0, relation="definition")
                for chunk in self.chunks
                if str(chunk.get("symbol")) in exact_symbols
                and chunk.get("kind") == "code"
            ]
            candidates.sort(
                key=lambda item: (
                    item.chunk["end"] - item.chunk["start"],
                    item.chunk["path"],
                )
            )
            return candidates[:limit]
        # Busca exata textual antes de qualquer embedding (símbolos sem AST).
        if identifiers and (not behavior_query or asks_callers):
            exact_hits = []
            handles = {identifier.lower() for identifier in identifiers}
            for index, blob in enumerate(self._blobs()):
                if any(handle in blob for handle in handles):
                    chunk = self.chunks[index]
                    relation = (
                        "definition"
                        if str(chunk.get("symbol") or "") in identifiers
                        else ("caller" if asks_callers else "match")
                    )
                    exact_hits.append(
                        CodeHit(chunk=chunk, score=5.0, relation=relation)
                    )
            if exact_hits:
                exact_hits.sort(
                    key=lambda item: (
                        0 if item.chunk.get("kind") == "code" else 1,
                        0 if item.chunk.get("symbol") else 1,
                        item.chunk["path"],
                        item.chunk["start"],
                    )
                )
                return exact_hits[:limit]
        scored: list[CodeHit] = []
        if behavior_query:
            terms = (
                query_tokens | concept_terms(query)
            ) - {"blender", "onde", "inicia", "inicalizado", "inicializado"}
            body_blobs = self._blobs(body_only=True)
            ranked = []
            for index, blob in enumerate(body_blobs):
                overlap = sum(1 for term in terms if term in blob)
                if overlap:
                    ranked.append((overlap, index))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            candidate_indices = [index for _, index in ranked[:800]]
        else:
            candidate_indices = range(len(self.chunks))
        for index in candidate_indices:
            chunk = self.chunks[index]
            text = str(chunk.get("text", ""))
            haystack = normalize_text(
                " ".join(
                    str(part) for part in (
                        chunk.get("path", ""), chunk.get("symbol") or "", text
                    ) if part
                )
            )
            score = 0.0
            relation = "match"
            chunk_symbol = str(chunk.get("symbol") or "")
            if identifiers and not behavior_query and not any(
                normalize_text(identifier) in haystack for identifier in identifiers
            ):
                continue
            if path_hint and path_hint.lower() in chunk["path"].lower():
                score += 3.0
            if explicit_symbol and explicit_symbol == chunk_symbol:
                score += 4.0
                relation = "definition"
            elif chunk_symbol and chunk_symbol in identifiers:
                score += 4.0
                relation = "definition"
            elif explicit_symbol and explicit_symbol in text:
                score += 2.5
                relation = "usage"
            if normalized and normalized in haystack:
                score += 2.0
            overlap = query_tokens & set(tokenize(haystack))
            if overlap:
                score += 1.4 * len(overlap) / max(1, len(query_tokens))
            score += 1.2 * max(0.0, cosine(query_vector, self._embedding(index)))
            if chunk.get("kind") == "test" and any(
                cue in normalized for cue in ("teste", "test", "falha", "bug")
            ):
                score += 0.35
            if chunk.get("kind") == "doc" and any(
                cue in normalized for cue in ("document", "readme", "como usar")
            ):
                score += 0.35
            if chunk.get("kind") == "code":
                if behavior_query:
                    score += 1.15
                    if chunk.get("symbol"):
                        score += 0.35
                    if (
                        any(
                            cue in normalized
                            for cue in ("inicia", "start", "servidor", "server", "desenvolvimento")
                        )
                        and str(chunk.get("symbol") or "") in {
                            "run", "serve", "serve_forever", "main", "start", "start_server"
                        }
                    ):
                        score += 1.4
                elif not any(
                    cue in normalized
                    for cue in ("document", "readme", "como usar", "teste")
                ):
                    score += 0.45
            elif chunk.get("kind") == "doc" and behavior_query:
                score -= 0.45
            if score > 0:
                scored.append(CodeHit(chunk=chunk, score=score, relation=relation))
        scored.sort(key=lambda item: (-item.score, item.chunk["path"], item.chunk["start"]))
        selected = scored[:limit]
        # Segunda busca dirigida: se a pergunta pede quem chama/usa, procura o símbolo encontrado.
        asks_callers = any(
            cue in normalized for cue in ("quem chama", "chama", "chamado", "usado", "usa", "referencia", "referência")
        )
        if asks_callers and selected:
            first_symbol = explicit_symbol or next(
                (
                    identifier
                    for identifier in identifiers
                    if any(chunk.get("symbol") == identifier for chunk in self.chunks)
                ),
                None,
            ) or next(
                (item.chunk.get("symbol") for item in selected if item.chunk.get("symbol")),
                None,
            )
            if first_symbol:
                seen = {item.chunk["id"] for item in selected}
                for item in selected:
                    if (
                        first_symbol in str(item.chunk.get("text", ""))
                        and item.chunk.get("symbol") != first_symbol
                    ):
                        item.relation = "caller"
                for item in scored:
                    if item.chunk["id"] in seen:
                        continue
                    if first_symbol in str(item.chunk.get("text", "")):
                        selected.append(
                            CodeHit(chunk=item.chunk, score=item.score + 1.0, relation="caller")
                        )
                        seen.add(item.chunk["id"])
                    if len(selected) >= limit + 2:
                        break
        return selected

    def context(
        self,
        query: str,
        *,
        max_chunks: int = 4,
        max_tokens: int = 260,
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        hits = self.search(query, limit=max_chunks)
        lines = [
            "[CÓDIGO INDEXADO]",
            f"Repositório: {self.repo_path}",
            f"Commit: {self.commit}",
            f"Licença: {(self.license or {}).get('name', 'não identificada')}",
        ]
        sources = []
        gaps = []
        used = 0
        for hit in hits:
            chunk = hit.chunk
            header = (
                f"[{chunk['id']}] {chunk['path']}:{chunk['start']}-{chunk['end']} "
                f"@ {chunk.get('commit', self.commit)[:12]}"
            )
            body = str(chunk.get("text", ""))
            block = header + "\n" + body
            estimate = len(block.split())
            if used + estimate > max_tokens and sources:
                break
            used += estimate
            lines.append(block)
            sources.append(hit.as_dict())
            if chunk.get("kind") == "comment":
                continue
            for comment in self.chunks:
                if (
                    comment.get("kind") != "comment"
                    or comment.get("path") != chunk.get("path")
                ):
                    continue
                if not (
                    (comment.get("symbol") and comment.get("symbol") == chunk.get("symbol"))
                    or (
                        comment.get("start", 0) >= chunk.get("start", 0)
                        and comment.get("end", 0) <= chunk.get("end", 0)
                    )
                ):
                    continue
                divergence = _divergence(str(comment.get("text", "")), body)
                if divergence:
                    source = {
                        "id": comment["id"],
                        "path": comment["path"],
                        "start": comment["start"],
                        "end": comment["end"],
                        "symbol": comment.get("symbol"),
                        "kind": "comment",
                        "commit": comment.get("commit"),
                        "score": 1.0,
                        "relation": "comment",
                        "text": comment.get("text", ""),
                        "divergence": divergence,
                    }
                    sources.append(source)
                    lines.append(
                        f"[{comment['id']}] {comment['path']}:{comment['start']}-"
                        f"{comment['end']} @ {comment.get('commit', self.commit)[:12]}\\n"
                        f"Comentário: {comment.get('text', '')}\n"
                        f"Divergência: {divergence}"
                    )
                break
        if not hits:
            gaps.append("trecho de código")
        if any(cue in normalize_text(query) for cue in ("quem chama", "chama", "usado", "usa")) and not any(
            hit.relation in {"caller", "usage"} for hit in hits
        ):
            gaps.append("quem chama/usa o símbolo encontrado")
        return "\n".join(lines), sources, gaps


def code_question(text: str) -> bool:
    normalized = normalize_text(text)
    if any(cue in normalized for cue in ("onde paramos", "retomar", "como estamos")):
        return False
    cues = (
        "onde", "implementad", "função", "funcao", "código", "codigo", "quem chama",
        "chama", "arquivo", "linha", "símbolo", "simbolo", "classe", "método",
        "metodo", "repositório", "repositorio", "commit", "definição", "definicao",
    )
    return any(cue in normalized for cue in cues)


def answer_from_index(project: Mapping[str, Any], query: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    index = CodeIndex.load(project)
    if index is None:
        return (
            "Ainda não há código indexado neste projeto. Associe o repositório e importe um commit.",
            [],
            ["índice de código"],
        )
    context, sources, gaps = index.context(query)
    lines = ["Aqui está o que encontrei no código indexado:", ""]
    for source in sources:
        symbol = f" ({source['symbol']})" if source.get("symbol") else ""
        lines.append(
            f"- `{source['path']}:{source['start']}-{source['end']}`{symbol} "
            f"@ {source['commit'][:12]} — [{source['relation']}]"
        )
        if source.get("divergence"):
            lines.append(f"  Divergência com o código: {source['divergence']}")
    if gaps:
        lines.append("Lacunas: " + "; ".join(gaps))
    lines.append("")
    lines.append(context)
    return "\n".join(lines), sources, gaps


def import_status(project: Mapping[str, Any]) -> dict[str, Any]:
    meta = project.get("code_index")
    if not meta:
        return {"indexed": False}
    return {"indexed": True, **meta}
