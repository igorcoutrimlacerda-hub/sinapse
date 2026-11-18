"""Importação, índice e busca de código-fonte local por projeto."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from sinapse.code_index import CodeIndex, answer_from_index, code_question, import_repository
from sinapse.project_copilot import ProjectStore


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def make_repo(base: Path, name: str) -> Path:
    repo = base / name
    repo.mkdir(parents=True)
    git(repo, "init")
    git(repo, "config", "user.email", "teste@example.com")
    git(repo, "config", "user.name", "Teste")
    return repo


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


class CodeIndexTest(unittest.TestCase):
    def test_import_search_definition_caller_and_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo = make_repo(base, "repo-a")
            (repo / "src").mkdir()
            (repo / "src" / "token.py").write_text(
                "# Retorna o token em maiúsculas.\n"
                "def resolve_token(value):\n"
                "    \"\"\"Normaliza o token do navegador.\"\"\"\n"
                "    return value.strip().lower()\n\n"
                "def use_token(raw):\n"
                "    return resolve_token(raw)\n",
                encoding="utf-8",
            )
            (repo / "README.md").write_text(
                "# Token\n\nResolve tokens do navegador. A aplicação inicia o servidor de desenvolvimento.\n",
                encoding="utf-8",
            )
            (repo / "src" / "server.py").write_text(
                "def run(host=None):\n"
                "    \"\"\"Inicia o servidor de desenvolvimento da aplicação.\"\"\"\n"
                "    return host\n",
                encoding="utf-8",
            )
            (repo / "LICENSE").write_text("MIT License\n\nPermission is hereby granted.", encoding="utf-8")
            (repo / "secret.env").write_text("API_KEY=sk-ficticia-nao-indexar-123456", encoding="utf-8")
            (repo / "image.bin").write_bytes(b"\x00\x01\x02")
            commit = commit_all(repo, "initial")

            store = ProjectStore(base / "projects")
            project = store.create("Code A")
            project["repo_path"] = str(repo)
            store.save(project)
            status = import_repository(store, project["id"])
            self.assertEqual(status["commit"], commit)
            self.assertEqual(status["license"]["name"], "MIT")
            self.assertNotIn("secret.env", json.dumps(status))
            loaded = store.load(project["id"])
            index = CodeIndex.load(loaded)
            self.assertIsNotNone(index)
            self.assertTrue(code_question("Onde resolve_token é implementado?"))
            self.assertFalse(code_question("Onde paramos?"))

            exact = index.search("resolve_token")
            self.assertIn("src/token.py", exact[0].chunk["path"])
            self.assertEqual(exact[0].chunk["symbol"], "resolve_token")

            natural = index.search("onde o token do navegador é normalizado")
            self.assertIn("src/token.py", natural[0].chunk["path"])
            behavior = index.search("onde a aplicação inicia o servidor de desenvolvimento")
            self.assertEqual(behavior[0].chunk["symbol"], "run")
            self.assertIn("src/server.py", behavior[0].chunk["path"])

            callers = index.search("quem chama resolve_token")
            relations = {hit.relation for hit in callers}
            self.assertIn("definition", relations)
            self.assertIn("caller", relations)
            self.assertTrue(any("use_token" in (hit.chunk.get("text") or "") for hit in callers))

            answer, sources, gaps = answer_from_index(loaded, "Onde resolve_token é implementado?")
            self.assertIn("src/token.py", answer)
            self.assertTrue(sources)
            self.assertEqual(gaps, [])
            self.assertIn("commit", sources[0])
            self.assertTrue(any(chunk.get("kind") == "comment" for chunk in index.chunks))
            divergent, divergent_sources, _ = answer_from_index(
                loaded, "Por que resolve_token retorna maiúsculas?"
            )
            self.assertIn("Divergência", divergent)
            self.assertTrue(any(source.get("divergence") for source in divergent_sources))

            from sinapse.integration import ProjectMemory

            memory = ProjectMemory(
                repo, store=store, project_id=project["id"], environ={}
            )
            resumed = memory.resume("Onde resolve_token é implementado?")
            self.assertTrue(resumed["code_sources"])
            self.assertIn("src/token.py", resumed["formatted"])
            self.assertIn("@", resumed["formatted"])

    def test_same_symbol_is_isolated_and_update_tracks_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo_a = make_repo(base, "repo-a")
            repo_b = make_repo(base, "repo-b")
            (repo_a / "main.py").write_text(
                "def calculate():\n    return 1\n", encoding="utf-8"
            )
            (repo_b / "main.py").write_text(
                "def calculate():\n    return 2\n", encoding="utf-8"
            )
            commit_all(repo_a, "a")
            commit_all(repo_b, "b")

            store = ProjectStore(base / "projects")
            project_a = store.create("A")
            project_a["repo_path"] = str(repo_a)
            store.save(project_a)
            project_b = store.create("B")
            project_b["repo_path"] = str(repo_b)
            store.save(project_b)
            import_repository(store, project_a["id"])
            import_repository(store, project_b["id"])
            index_a = CodeIndex.load(store.load(project_a["id"]))
            index_b = CodeIndex.load(store.load(project_b["id"]))
            hit_a = index_a.search("calculate")[0]
            hit_b = index_b.search("calculate")[0]
            self.assertIn("return 1", hit_a.chunk["text"])
            self.assertIn("return 2", hit_b.chunk["text"])
            self.assertNotEqual(hit_a.chunk["commit"], hit_b.chunk["commit"])

            (repo_a / "main.py").write_text(
                "def calculate():\n    return 10\n\n"
                "def extra():\n    return 'novo'\n",
                encoding="utf-8",
            )
            (repo_a / "old.py").write_text("def old():\n    return 'antigo'\n", encoding="utf-8")
            commit_all(repo_a, "second")
            (repo_a / "old.py").unlink()
            new_commit = commit_all(repo_a, "remove old")
            import_repository(store, project_a["id"])
            updated = CodeIndex.load(store.load(project_a["id"]))
            self.assertEqual(updated.commit, new_commit)
            self.assertIn("return 10", updated.search("calculate")[0].chunk["text"])
            self.assertTrue(any("extra" == hit.chunk.get("symbol") for hit in updated.search("extra")))
            self.assertFalse(any(hit.chunk["path"] == "old.py" for hit in updated.search("old")))

    def test_missing_index_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = ProjectStore(base / "projects")
            project = store.create("Sem código")
            answer, sources, gaps = answer_from_index(project, "Onde a função é definida?")
            self.assertIn("não há código indexado", answer.lower())
            self.assertEqual(sources, [])
            self.assertEqual(gaps, ["índice de código"])


if __name__ == "__main__":
    unittest.main()
