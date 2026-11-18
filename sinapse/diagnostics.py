"""Logs técnicos do Farol, separados da memória conversacional.

Cada projeto tem um arquivo JSONL próprio em ``<projects-root>/../diagnostics``.
Os eventos não entram na RAM, no prompt nem na busca Sinapse. A consulta é
somente leitura e por projeto.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DiagnosticLog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        safe = "".join(
            char for char in project_id if char.isalnum() or char in ("-", "_")
        )
        return self.root / f"{safe or 'projeto'}.jsonl"

    def record(
        self,
        project_id: str,
        stage: str,
        *,
        status: str = "ok",
        request_id: str | None = None,
        latency_ms: float | None = None,
        error_code: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "schema": "farol.diag.v1",
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "request_id": request_id or f"req_{uuid.uuid4().hex[:12]}",
            "project_id": project_id,
            "ts": _now(),
            "stage": stage,
            "status": status,
            "latency_ms": latency_ms,
            "error_code": error_code,
            "details": dict(details or {}),
        }
        with self._path(project_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def events(
        self, project_id: str, *, since: str | None = None
    ) -> list[dict[str, Any]]:
        path = self._path(project_id)
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if since and str(event.get("ts", "")) < since:
                continue
            events.append(event)
        return events

    def delete(self, project_id: str) -> None:
        path = self._path(project_id)
        if path.exists():
            path.unlink()

    def projects(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.jsonl"))
