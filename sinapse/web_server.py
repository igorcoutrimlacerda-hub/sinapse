"""Servidor web local mínimo do Farol.

Interface HTTP baseada apenas na biblioteca padrão. A chave do DeepSeek fica no
processo do servidor e nunca é enviada ao navegador.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from .config import load_config
from .code_index import (
    CodeIndex,
    answer_from_index,
    code_question,
    import_repository,
    import_status,
)
from .farol_settings import DEFAULT_PATH, FarolSettings, redact_key
from .project_copilot import ProjectStore, chat_with_deepseek, local_answer

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 64 * 1024


def _payload(store: ProjectStore, project: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "project": {
            "id": project["id"],
            "name": project["name"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
            "repo_path": project.get("repo_path"),
            "semantic": bool(project.get("semantic", False)),
            "code_index": import_status(project),
        },
        "messages": project["messages"],
        "resume": store.resume(project),
        "dashboard": store.dashboard(project),
    }


def _project_summary(store: ProjectStore) -> dict[str, Any]:
    return {
        "projects": store.list(),
        "active": store.active_id(),
    }


class FarolHandler(BaseHTTPRequestHandler):
    server_version = "Farol/0.1"

    @property
    def store(self) -> ProjectStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def config(self) -> Mapping[str, Any]:
        return self.server.config  # type: ignore[attr-defined]

    @property
    def settings(self) -> FarolSettings:
        return self.server.settings  # type: ignore[attr-defined]

    def _simulated_client(self, key: str | None):
        factory = getattr(self.server, "client_factory", None)
        if factory is None or not key:
            return None
        return factory(key)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[farol] " + format % args + "\n")

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> Mapping[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY:
            raise ValueError("Requisição grande demais")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("Corpo JSON inválido")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/settings":
                self._json(200, self.settings.status())
                return
            if path == "/api/projects":
                self._json(200, _project_summary(self.store))
                return
            if path.startswith("/api/projects/") and path.endswith("/code"):
                parts = [part for part in path.split("/") if part]
                if len(parts) == 4:
                    project = self.store.load(parts[2])
                    self._json(200, import_status(project))
                    return
            if "/code/chunk/" in path:
                parts = [part for part in path.split("/") if part]
                if len(parts) == 6 and parts[3] == "code" and parts[4] == "chunk":
                    project = self.store.load(parts[2])
                    index = CodeIndex.load(project)
                    if index is None:
                        self._error(404, "Projeto sem índice de código")
                        return
                    chunk = next(
                        (item for item in index.chunks if item.get("id") == parts[5]),
                        None,
                    )
                    if chunk is None:
                        self._error(404, "Trecho não encontrado")
                        return
                    self._json(200, chunk)
                    return
            if path.startswith("/api/projects/"):
                parts = [part for part in path.split("/") if part]
                # /api/projects/<id>
                if len(parts) == 3:
                    project = self.store.load(parts[2])
                    self._json(200, _payload(self.store, project))
                    return
                # /api/projects/<id>/messages/<message_id>
                if len(parts) == 5 and parts[3] == "messages":
                    project = self.store.load(parts[2])
                    message = self.store.show(parts[4], project)
                    self._json(200, message)
                    return
            self._serve_static(path)
        except ValueError as error:
            message = str(error)
            status = 403 if "restrita" in message else 404
            self._error(status, message)
        except Exception as error:  # pragma: no cover - proteção do servidor
            self._error(500, f"Erro interno: {error}")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            data = self._body()
            if path == "/api/settings/key":
                status = self.settings.save_key(
                    str(data.get("api_key", "")),
                    data.get("model"),
                )
                self._json(200, status)
                return
            if path == "/api/settings/remove":
                self._json(200, self.settings.remove_key())
                return
            if path == "/api/settings/test":
                active_key, _ = self.settings.active_key()
                self._json(
                    200,
                    self.settings.test_connection(
                        self._simulated_client(active_key)
                    ),
                )
                return
            if path == "/api/settings/model":
                self._json(
                    200, self.settings.set_model(str(data.get("model", "")))
                )
                return
            if path == "/api/projects":
                name = str(data.get("name", "")).strip()
                if not name:
                    self._error(400, "Informe o nome do projeto")
                    return
                project = self.store.create(name)
                self._json(201, _payload(self.store, project))
                return
            parts = [part for part in path.split("/") if part]
            if len(parts) == 4 and parts[3] == "select":
                project = self.store.select(parts[2])
                self._json(200, _payload(self.store, project))
                return
            if len(parts) == 5 and parts[3] == "code" and parts[4] == "import":
                status = import_repository(
                    self.store,
                    parts[2],
                    commit=data.get("commit") if data else None,
                )
                project = self.store.load(parts[2])
                payload = _payload(self.store, project)
                payload["code_index"] = status
                self._json(200, payload)
                return
            if len(parts) == 4 and parts[3] == "messages":
                text = str(data.get("text", "")).strip()
                if not text:
                    self._error(400, "Mensagem vazia")
                    return
                project = self.store.load(parts[2])
                reservation = self.store.consume_anticipation(parts[2], text)
                project = self.store.load(parts[2])
                self.store.add_message(project, text, author="user")
                reserved_ids: list[str] = []
                if (
                    reservation
                    and int(project.get("revision", 0))
                    == int(reservation.get("project_revision", -1))
                    and self.store.valid_anticipation(project, reservation)
                ):
                    reserved_ids = list(reservation.get("source_ids", ()))
                source = "local"
                warning = None
                if bool(data.get("deepseek")):
                    active_key = None
                    try:
                        active_key, _active_source = self.settings.active_key()
                        active_model = self.settings.model
                        answer = chat_with_deepseek(
                            project,
                            text,
                            self.config,
                            reserved_ids,
                            store=self.store,
                            api_key=active_key,
                            model=active_model,
                            client=self._simulated_client(active_key),
                        )
                        source = "deepseek"
                    except Exception as error:
                        warning = (
                            "DeepSeek indisponível; resposta local usada: "
                            + redact_key(str(error), active_key)
                        )
                        answer = local_answer(self.store, project, text)
                else:
                    answer = local_answer(self.store, project, text)
                code_sources = []
                if code_question(text):
                    _answer, code_sources, _gaps = answer_from_index(project, text)
                assistant = self.store.add_message(
                    project,
                    answer,
                    author="assistant",
                    source=source,
                    code_sources=code_sources,
                )
                payload = _payload(self.store, project)
                payload["assistant"] = assistant
                payload["source"] = source
                payload["warning"] = warning
                self._json(200, payload)
                self.store.start_anticipation(parts[2], text, self.config)
                return
            if len(parts) == 4 and parts[3] == "repo":
                project = self.store.load(parts[2])
                value = str(data.get("repo_path", "")).strip()
                if value:
                    candidate = Path(value).expanduser()
                    if not candidate.is_dir():
                        self._error(400, "Repositório não encontrado")
                        return
                    project["repo_path"] = str(candidate.resolve())
                else:
                    project.pop("repo_path", None)
                self.store.save(project)
                self.store.diagnostics.record(
                    parts[2],
                    "project.repo_linked",
                    details={"repo_path": project.get("repo_path")},
                )
                self._json(200, _payload(self.store, project))
                return
            if len(parts) == 4 and parts[3] == "semantic":
                project = self.store.load(parts[2])
                project["semantic"] = bool(data.get("enabled"))
                self.store.save(project)
                self.store.diagnostics.record(
                    parts[2],
                    "project.semantic_toggled",
                    details={"enabled": project["semantic"]},
                )
                self._json(200, _payload(self.store, project))
                return
            if len(parts) == 4 and parts[3] == "delete":
                deleted = self.store.delete(
                    parts[2],
                    str(data.get("confirm_id", "")),
                    str(data.get("confirm_name", "")),
                )
                payload = _project_summary(self.store)
                payload["deleted"] = deleted
                self._json(200, payload)
                return
            self._error(404, "Rota não encontrada")
        except ValueError as error:
            self._error(400, str(error))
        except Exception as error:  # pragma: no cover - proteção do servidor
            self._error(500, f"Erro interno: {error}")

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB_DIR / relative).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.exists():
            self._error(404, "Arquivo não encontrado")
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def create_server(
    root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    settings_path: str | Path = DEFAULT_PATH,
    *,
    client_factory=None,
    environ: Mapping[str, str] | None = None,
) -> ThreadingHTTPServer:
    """Cria o servidor sem bloquear; usado pela CLI e pelos testes."""
    store = ProjectStore(root)
    server = ThreadingHTTPServer((host, port), FarolHandler)
    server.store = store  # type: ignore[attr-defined]
    server.config = load_config()  # type: ignore[attr-defined]
    server.settings = FarolSettings(  # type: ignore[attr-defined]
        settings_path, config=server.config, environ=environ
    )
    server.client_factory = client_factory  # type: ignore[attr-defined]
    return server


def run_server(
    root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    settings_path: str | Path = DEFAULT_PATH,
    *,
    client_factory=None,
    environ: Mapping[str, str] | None = None,
) -> None:
    server = create_server(
        root,
        host,
        port,
        settings_path,
        client_factory=client_factory,
        environ=environ,
    )
    status = server.settings.status()  # type: ignore[attr-defined]
    print(f"Farol em http://{host}:{port}")
    print(f"Projetos: {Path(root).resolve()}")
    source = {
        "panel": "chave salva no painel",
        "environment": "DEEPSEEK_API_KEY",
        "none": "não configurado (modo local)",
    }[status["source"]]
    print(f"DeepSeek: {source}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
