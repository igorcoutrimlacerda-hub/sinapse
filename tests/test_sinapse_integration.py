"""Integração local do Sinapse com agentes (bridge e resolução por repo)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sinapse.integration import ProjectMemory, find_repo_root, stable_entry_id
from sinapse.project_copilot import ProjectStore


def make_repo(base: Path, name: str) -> Path:
    repo = base / name
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    return repo


class ProjectMemoryTest(unittest.TestCase):
    def make(self, base: Path, repo: Path, store: ProjectStore) -> ProjectMemory:
        return ProjectMemory(repo, store=store, environ={})

    def test_resume_links_repo_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "projeto-a")
            store = ProjectStore(base / "memory")
            memory = self.make(base, repo, store)

            first = memory.record("decision", "usar Python 3.13", entry_id="dec-py")
            self.assertFalse(first["duplicate"])
            again = memory.record("decision", "usar Python 3.13", entry_id="dec-py")
            self.assertTrue(again["duplicate"])
            self.assertEqual(len(memory.project_record()["messages"]), 1)

            other = ProjectMemory(repo, store=ProjectStore(base / "memory"), environ={})
            resumed = other.resume()
            self.assertEqual(resumed["decision"]["value"], "usar Python 3.13")
            self.assertEqual(resumed["decision"]["source_ids"], ["m1"])
            self.assertIn("usar Python 3.13 [m1]", resumed["formatted"])
            self.assertIn(resumed["project_id"], resumed["formatted"])

    def test_correction_supersedes_and_keeps_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "projeto-b")
            store = ProjectStore(base / "memory")
            memory = self.make(base, repo, store)

            memory.record(
                "decision",
                "preset Serum",
                entry_id="dec-preset",
                reason="teste aprovado",
            )
            memory.record(
                "correction",
                "preset Dunas",
                entry_id="dec-preset-dunas",
                supersedes="dec-preset",
            )
            resumed = memory.resume()
            self.assertEqual(resumed["decision"]["value"], "preset Dunas")
            self.assertEqual(resumed["corrections"][0]["from"], "preset Serum")
            self.assertEqual(resumed["corrections"][0]["to"], "preset Dunas")
            self.assertEqual(resumed["corrections"][0]["source_ids"], ["m1"])
            self.assertEqual(
                resumed["corrections"][0]["correction_source_ids"], ["m2"]
            )
            self.assertIn("preset Serum", resumed["sources"]["m1"]["text"])
            self.assertIn("preset Dunas", resumed["sources"]["m2"]["text"])

            # A correção vale na próxima retomada, vinda de um novo processo.
            reloaded = ProjectMemory(
                repo, store=ProjectStore(base / "memory"), environ={}
            )
            self.assertEqual(reloaded.resume()["decision"]["value"], "preset Dunas")

    def test_pending_closed_by_test_and_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "projeto-c")
            memory = self.make(base, repo, ProjectStore(base / "memory"))

            memory.record(
                "pending",
                "rodar suíte de integração",
                entry_id="pend-suite",
            )
            self.assertEqual(
                memory.resume()["next_step"]["text"], "rodar suíte de integração"
            )
            memory.record(
                "test",
                "suíte de integração passou",
                entry_id="test-suite",
                resolves="pend-suite",
            )
            resumed = memory.resume()
            self.assertEqual(resumed["last_verified"]["text"], "suíte de integração passou")
            self.assertIsNone(resumed["next_step"])
            self.assertEqual(resumed["pending"], [])

    def test_test_record_only_closes_pending_with_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "repo-pendencia")
            memory = self.make(base, repo, ProjectStore(base / "memory"))
            memory.record(
                "pending",
                "validar a exclusão de projeto no navegador",
                entry_id="pend-exclusao",
            )
            memory.record(
                "test",
                "auditoria de contexto no navegador e nos projetos",
                entry_id="test-auditoria",
            )
            state = memory.project_record()["state"]["pending"]
            self.assertEqual(state[0]["status"], "aberto")
            self.assertIsNone(state[0].get("completed_by"))
            memory.record(
                "test",
                "exclusão validada",
                entry_id="test-exclusao",
                resolves="pend-exclusao",
            )
            state = memory.project_record()["state"]["pending"]
            self.assertEqual(state[0]["status"], "concluido")
            self.assertEqual(state[0]["completed_by"], "m3")

    def test_search_isolated_by_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo_a = make_repo(base, "repo-a")
            repo_b = make_repo(base, "repo-b")
            store = ProjectStore(base / "memory")
            memory_a = self.make(base, repo_a, store)
            memory_b = self.make(base, repo_b, store)

            memory_a.record("decision", "banco SQLite no protótipo", entry_id="a-db")
            memory_b.record("decision", "banco Postgres em produção", entry_id="b-db")

            hits_a = memory_a.search("banco")
            hits_b = memory_b.search("banco")
            self.assertEqual([hit["entry_id"] for hit in hits_a["hits"]], ["a-db"])
            self.assertEqual([hit["entry_id"] for hit in hits_b["hits"]], ["b-db"])
            self.assertNotIn("Postgres", hits_a["message"])
            self.assertNotIn("SQLite", hits_b["message"])
            self.assertNotEqual(
                memory_a.resume()["project_id"], memory_b.resume()["project_id"]
            )

    def test_marker_and_bridge_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "repo-marcado")
            store = ProjectStore(base / "memory")
            target = store.create("Projeto escolhido")
            (repo / ".sinapse.json").write_text(
                '{"project": "' + target["id"] + '"}', encoding="utf-8"
            )
            memory = self.make(base, repo, store)
            self.assertEqual(memory.project_id(), target["id"])
            linked = memory.link()
            self.assertTrue((repo / ".sinapse.json").exists())
            self.assertIn(target["id"], linked["message"])

    def test_rejects_credentials_without_echo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "repo-segredo")
            memory = self.make(base, repo, ProjectStore(base / "memory"))
            secret = "sk-abcdefghijklmnopqrstuvwx"
            with self.assertRaises(ValueError) as raised:
                memory.record("note", f"chave {secret}")
            self.assertNotIn(secret, str(raised.exception))
            self.assertEqual(memory.project_record()["messages"], [])

    def test_stable_entry_id_is_deterministic(self) -> None:
        first = stable_entry_id("projeto", "decision", "usar Dunas")
        second = stable_entry_id("projeto", "decision", "  usar   Dunas ")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("decision-"))

    def test_find_repo_root_stops_at_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = make_repo(Path(directory), "raiz")
            nested = repo / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(find_repo_root(nested), repo.resolve())


if __name__ == "__main__":
    unittest.main()
