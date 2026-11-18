"""Farol para projetos de software: memória técnica, seleção e payload."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sinapse.config import load_config
from sinapse.deepseek import ChatCompletionResponse, load_deepseek_settings
from sinapse.integration import ProjectMemory
from sinapse.project_copilot import (
    ProjectStore,
    _prepare_chat_payload,
    chat_with_deepseek,
    format_resume,
    local_answer,
)
from sinapse.software_context import classify_request, render_state, select_state


class Capture:
    simulated = True
    name = "capture"

    def __init__(self) -> None:
        self.requests = []

    def complete(self, body, timeout=60.0):
        self.requests.append(json.loads(json.dumps(body)))
        return ChatCompletionResponse(
            content="resposta simulada", model=body.get("model", "")
        )


def make_repo(base: Path, name: str) -> Path:
    repo = base / name
    repo.mkdir(parents=True)
    (repo / ".git").mkdir()
    return repo


def software_memory(base: Path, name: str = "login-repo"):
    store = ProjectStore(base / "projects")
    repo = make_repo(base, name)
    memory = ProjectMemory(repo, store=store, environ={})
    memory.record("goal", "Migrar autenticação para sessão opaca.", entry_id="goal-auth")
    memory.record("task", "Implementar revogação de sessão.", entry_id="task-auth")
    memory.record(
        "architecture",
        "Usar JWT no login.",
        reason="API stateless.",
        files=["src/auth.py"],
        restrictions=["sem cookies"],
        entry_id="arch-jwt",
    )
    memory.record(
        "bug",
        "Login falha ao expirar sessão.",
        symptom="401 após 15 min",
        files=["src/auth.py", "tests/test_login.py"],
        command="pytest tests/test_login.py",
        entry_id="bug-login",
    )
    memory.record(
        "correction",
        "Usar sessão opaca no login.",
        supersedes="arch-jwt",
        reason="precisamos revogar tokens",
        files=["src/auth.py", "src/session.py"],
        restrictions=["sem JWT"],
        entry_id="arch-opaca",
    )
    memory.record(
        "test",
        "Teste do login com sessão opaca passou.",
        command="pytest tests/test_login.py",
        reported_by="opencode",
        report_source="relatório ses_login",
        resolves="bug-login",
        entry_id="test-login",
    )
    memory.record(
        "test",
        "Testar expiração em 15 min.",
        status="suggested",
        command="pytest tests/test_session.py",
        entry_id="test-expiracao",
    )
    memory.record(
        "pending",
        "Validar revogação no logout.",
        command="pytest tests/test_session.py",
        entry_id="pend-logout",
    )
    return store, memory, store.load(memory.project_id())


class SoftwareContextTest(unittest.TestCase):
    def test_resume_technical_fields_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store, memory, project = software_memory(base)
            resumed = store.resume(project, "Onde paramos?")
            self.assertEqual(resumed["current_task"], "Implementar revogação de sessão.")
            self.assertEqual(resumed["current_task_source_ids"], ["m2"])
            self.assertEqual(resumed["decision"]["value"], "Usar sessão opaca no login.")
            self.assertEqual(resumed["reason"], "precisamos revogar tokens")
            self.assertEqual(resumed["decision"]["files"], ["src/auth.py", "src/session.py"])
            self.assertEqual(resumed["decision"]["restrictions"], ["sem JWT"])
            self.assertEqual(resumed["last_test"]["source_ids"], ["m6"])
            self.assertEqual(resumed["last_test"]["command"], "pytest tests/test_login.py")
            self.assertEqual(
                resumed["last_test"]["report_source"], "relatório ses_login"
            )
            self.assertEqual(resumed["next_step"]["text"], "Validar revogação no logout.")
            self.assertEqual(resumed["goal_source_ids"], ["m1"])
            text = format_resume(resumed)
            for label in (
                "Tarefa atual",
                "Decisão técnica",
                "Último teste observado",
                "Próximo passo",
                "pytest tests/test_login.py",
            ):
                self.assertIn(label, text)

    def test_multi_part_uses_decision_and_pending_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store, _memory, project = software_memory(base)
            query = "Por que escolhemos sessão opaca no login e qual teste falta?"
            fields = select_state(project, query)
            self.assertEqual(fields["decision"]["value"], "Usar sessão opaca no login.")
            self.assertEqual(
                [item.get("entry_id") for item in fields["pending"]],
                ["test-expiracao", "pend-logout"],
            )
            self.assertEqual(fields["decision"]["reason"], "precisamos revogar tokens")
            self.assertFalse(fields["gaps"])

            answer = local_answer(store, project, query)
            self.assertIn("sessão opaca no login", answer)
            self.assertIn("Teste", answer)
            self.assertNotIn("Usar JWT no login.", answer)

            resumed = store.resume(project, "Onde paramos?")
            self.assertFalse(resumed["show_corrections"])
            self.assertNotIn("Correções registradas", format_resume(resumed))

            changed = select_state(project, "O que mudou na decisão de login?")
            self.assertTrue(changed["corrections"])
            self.assertIn("Usar JWT no login.", json.dumps(changed, ensure_ascii=False))

    def test_semantic_gap_recovers_different_vocabulary_and_keeps_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = ProjectStore(base / "projects")
            repo = make_repo(base, "semantic-repo")
            memory = ProjectMemory(repo, store=store, environ={})
            memory.record(
                "architecture",
                "Adotamos JWT no cliente.",
                reason="o time já conhecia o formato.",
                entry_id="sem-jwt",
            )
            memory.record(
                "correction",
                "Adotamos sessões opacas no servidor.",
                supersedes="sem-jwt",
                reason="precisamos revogar tokens no navegador.",
                entry_id="sem-opaca",
            )
            for text in (
                "Revisamos os textos de ajuda.",
                "O layout da tela inicial mudou.",
                "Atualizamos a documentação do módulo.",
                "Limpamos o cache da máquina de testes.",
            ):
                store.add_message(store.load(memory.project_id()), text, author="user")
            project = store.load(memory.project_id())
            query = "Por que abandonamos o cookie no cliente?"

            lexical = select_state(project, query)
            self.assertIsNone(lexical["decision"])
            self.assertIn("decisão/motivo", lexical["gaps"])

            semantic = select_state(project, query, semantic=True)
            self.assertEqual(semantic["decision"]["value"], "Adotamos sessões opacas no servidor.")
            self.assertEqual(semantic["decision"]["source_ids"], ["m2"])
            self.assertNotIn("m1", semantic["decision"]["source_ids"])
            self.assertFalse(semantic["gaps"])

            revoked = select_state(project, "Por que trocamos o JWT no cliente?", semantic=True)
            self.assertEqual(revoked["decision"]["source_ids"], ["m2"])
            self.assertNotIn("m1", revoked["decision"]["source_ids"])

            missing = select_state(
                project, "Por que removemos a fila de eventos?", semantic=True
            )
            self.assertTrue(missing["gaps"])

    def test_link_traversal_adds_reason_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store, _memory, project = software_memory(base)
            store.add_message(
                store.load(project["id"]),
                "Motivo original registrado em ata: precisamos revogar tokens no navegador.",
                author="user",
            )
            project = store.load(project["id"])
            reason_message = project["messages"][-1]["id"]
            for decision in project["state"]["decisions"]:
                if decision.get("status") == "vigente":
                    decision["reason_source_ids"] = [reason_message]
            store.save(project)
            fields = select_state(project, "Por que a sessão opaca foi escolhida?")
            self.assertEqual(fields["decision"]["value"], "Usar sessão opaca no login.")
            linked_ids = [
                source for item in fields["related"] for source in item.get("source_ids", ())
            ]
            self.assertIn(reason_message, linked_ids)
            rendered = render_state(fields)
            self.assertIn(reason_message, rendered)
            self.assertIn("Motivo original", rendered)

    def test_selector_respects_subject_and_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _store, _memory, project = software_memory(base)

            current = select_state(project, "Qual é a decisão vigente sobre o login?")
            self.assertEqual(current["decision"]["value"], "Usar sessão opaca no login.")
            self.assertNotIn("Usar JWT no login.", json.dumps(current))

            other = select_state(project, "O que decidimos sobre o cache?")
            self.assertIsNone(other["decision"])

            bug = select_state(project, "O bug do login")
            self.assertEqual(bug["bugs"][0]["text"], "Login falha ao expirar sessão.")
            self.assertEqual(bug["bugs"][0]["status"], "corrigido")
            self.assertIn("m6", [source for test in bug["tests"] for source in test["source_ids"]])

            goal = select_state(project, "Qual é o objetivo?")
            self.assertEqual(goal["goal"]["text"], "Migrar autenticação para sessão opaca.")
            self.assertEqual(goal["goal"]["source_ids"], ["m1"])

            self.assertEqual(classify_request("Oi, tudo bem?"), "casual")
            self.assertEqual(classify_request("Onde paramos?"), "resume")
            self.assertEqual(classify_request("O bug do login"), "bug")

    def test_bug_payload_and_greeting_without_technical_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store, _memory, project = software_memory(base)
            client = Capture()
            chat_with_deepseek(project, "O bug do login", load_config(), store=store, client=client)
            body = client.requests[0]
            text = json.dumps(body, ensure_ascii=False)
            self.assertIn("ESTADO PERSISTIDO", text)
            self.assertIn("Login falha ao expirar sessão.", text)
            self.assertIn("src/auth.py", text)
            self.assertIn("pytest tests/test_login.py", text)
            self.assertIn("[m4]", text)
            self.assertIn("relatório ses_login", text)
            messages = [message["content"] for message in body["messages"]]
            self.assertEqual(sum(content.count("O bug do login") for content in messages), 1)

            greeting = Capture()
            chat_with_deepseek(project, "Oi, tudo bem?", load_config(), store=store, client=greeting)
            greeting_text = json.dumps(greeting.requests[0], ensure_ascii=False)
            self.assertNotIn("ESTADO PERSISTIDO", greeting_text)
            self.assertNotIn("Login falha", greeting_text)
            self.assertNotIn("pytest tests/test_login.py", greeting_text)
            self.assertEqual(
                sum(
                    message["content"].count("Oi, tudo bem?")
                    for message in greeting.requests[0]["messages"]
                ),
                1,
            )

    def test_credentials_redacted_in_payload_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = ProjectStore(base / "projects")
            project = store.create("Credencial")
            secret = "sk-teste-software-1234567890"
            store.add_message(project, f"A chave fictícia é {secret}.", author="user")
            store.add_message(project, "O bug usa a chave acima.", author="user")
            project = store.load()
            client = Capture()
            chat_with_deepseek(project, "Qual é a chave?", load_config(), store=store, client=client)
            payload = json.dumps(client.requests[0], ensure_ascii=False)
            self.assertNotIn(secret, payload)
            self.assertIn("[credencial removida]", payload)
            events = store.diagnostics.events(project["id"])
            self.assertNotIn(secret, json.dumps(events, ensure_ascii=False))

    def test_budget_trims_optional_and_raises_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store, _memory, project = software_memory(base)
            config = load_config()
            settings = replace(
                load_deepseek_settings(config),
                context_window=120,
                output_reserve=40,
                max_tokens=40,
            )
            with self.assertRaises(ValueError) as raised:
                _prepare_chat_payload(project, "Onde paramos?", settings)
            self.assertIn("Contexto mínimo excede", str(raised.exception))

            settings = replace(
                load_deepseek_settings(config),
                context_window=520,
                output_reserve=100,
                max_tokens=100,
            )
            body, info = _prepare_chat_payload(project, "Onde paramos?", settings)
            self.assertLessEqual(info["tokens"], info["capacity"])
            self.assertLessEqual(len(info["recent_ids"]), 4)

    def test_suggested_test_is_pending_and_reported_test_is_attributed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _store, memory, project = software_memory(base)
            pending = project["state"]["pending"]
            suggested = next(item for item in pending if item["kind"] == "test-suggested")
            self.assertEqual(suggested["status"], "aberto")
            self.assertEqual(suggested["command"], "pytest tests/test_session.py")
            self.assertFalse(
                any(
                    item.get("entry_id") == "test-expiracao"
                    for item in project["state"]["completed"]
                )
            )
            completed = project["state"]["completed"]
            reported = next(item for item in completed if item.get("entry_id") == "test-login")
            self.assertEqual(reported["reported_by"], "opencode")
            self.assertEqual(reported["report_source"], "relatório ses_login")
            self.assertEqual(reported["command"], "pytest tests/test_login.py")
            bug = next(item for item in project["state"]["bugs"] if item.get("entry_id") == "bug-login")
            self.assertEqual(bug["status"], "corrigido")

    def test_same_bug_name_is_isolated_between_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = ProjectStore(base / "projects")
            repo_a = make_repo(base, "repo-a")
            repo_b = make_repo(base, "repo-b")
            memory_a = ProjectMemory(repo_a, store=store, environ={})
            memory_b = ProjectMemory(repo_b, store=store, environ={})
            memory_a.record(
                "bug",
                "Login falha ao expirar sessão.",
                symptom="401 no cliente A",
                entry_id="bug-a",
            )
            memory_b.record(
                "bug",
                "Login falha ao expirar sessão.",
                symptom="timeout no cliente B",
                entry_id="bug-b",
            )
            bug_a = select_state(memory_a.project_record(), "O bug do login")["bugs"]
            bug_b = select_state(memory_b.project_record(), "O bug do login")["bugs"]
            self.assertIn("cliente A", json.dumps(bug_a, ensure_ascii=False))
            self.assertNotIn("cliente B", json.dumps(bug_a, ensure_ascii=False))
            self.assertIn("cliente B", json.dumps(bug_b, ensure_ascii=False))
            self.assertNotIn("cliente A", json.dumps(bug_b, ensure_ascii=False))
            self.assertNotEqual(memory_a.project_id(), memory_b.project_id())

    def test_architecture_components_coexist_and_only_same_component_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _store, memory, project = software_memory(base)
            memory.record(
                "architecture",
                "Usar Redis para cache de sessão.",
                component="cache",
                entry_id="arch-cache",
            )
            memory.record(
                "architecture",
                "Usar fila local para eventos.",
                component="fila",
                entry_id="arch-fila",
            )
            project = memory.project_record()
            vigentes = {
                item.get("component")
                for item in project["state"]["decisions"]
                if item.get("status") == "vigente"
            }
            self.assertTrue({"cache", "fila"}.issubset(vigentes))
            memory.record(
                "correction",
                "Usar Postgres para cache de sessão.",
                component="cache",
                supersedes="arch-cache",
                entry_id="arch-cache-pg",
            )
            project = memory.project_record()
            cache = [
                item for item in project["state"]["decisions"]
                if item.get("component") == "cache"
            ]
            fila = [
                item for item in project["state"]["decisions"]
                if item.get("component") == "fila"
            ]
            self.assertEqual(
                [item["value"] for item in cache if item["status"] == "vigente"],
                ["Usar Postgres para cache de sessão."],
            )
            self.assertEqual(
                [item["value"] for item in fila if item["status"] == "vigente"],
                ["Usar fila local para eventos."],
            )

    def test_project_semantic_flag_enables_local_gap_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            store = ProjectStore(base / "projects")
            repo = make_repo(base, "semantic-local")
            memory = ProjectMemory(repo, store=store, environ={})
            memory.record(
                "architecture",
                "Adotamos JWT no cliente.",
                entry_id="local-jwt",
            )
            memory.record(
                "correction",
                "Adotamos sessões opacas no servidor.",
                supersedes="local-jwt",
                reason="precisamos revogar tokens no navegador.",
                entry_id="local-opaca",
            )
            for text in (
                "Revisamos os textos de ajuda.",
                "O layout da tela inicial mudou.",
                "Atualizamos a documentação do módulo.",
                "Limpamos o cache da máquina de testes.",
            ):
                store.add_message(store.load(memory.project_id()), text, author="user")
            project = store.load(memory.project_id())
            query = "Por que abandonamos o cookie no cliente?"
            self.assertNotIn("sessões opacas", local_answer(store, project, query))
            project["semantic"] = True
            store.save(project)
            answer = local_answer(store, project, query)
            self.assertIn("sessões opacas", answer)

    def test_legacy_music_project_still_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Norte")
            for text in (
                "Comecei o Norte com o preset Lume.",
                "Troquei para o preset Safira porque a CPU do Lume ficava alta.",
                "O teste do refrão ficou concluído.",
                "Se a CPU ficar abaixo de 30%, exportar em 48 kHz.",
                "Falta testar a ponte em 48 kHz.",
            ):
                store.add_message(store.load(), text)
            project = store.load()
            resumed = store.resume(project, "Onde paramos?")
            self.assertEqual(resumed["decision"]["value"], "Safira")
            self.assertEqual(resumed["reason"], "a CPU do Lume ficava alta")
            self.assertEqual(resumed["last_test"]["text"], "O teste do refrão ficou concluído.")
            self.assertIn("Tarefa atual", format_resume(resumed))
            cpu = local_answer(store, project, "A CPU ficou em 27%.")
            self.assertIn("27%", cpu)


if __name__ == "__main__":
    unittest.main()
