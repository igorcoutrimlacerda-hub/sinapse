"""MVP do copiloto de projetos: persistência, isolamento e retomada."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sinapse.project_copilot import ProjectStore, format_resume


class ProjectCopilotTest(unittest.TestCase):
    def test_create_chat_restart_resume_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            norte = store.create("Norte")
            store.create("Norte II")
            store.create("Vale")

            store.select("norte")
            for text in (
                "Comecei o projeto Norte com o preset Lume no bass.",
                "No Norte, troquei para o preset Safira porque a CPU do Lume ficava alta.",
                "O teste do Safira no refrão ficou concluído.",
                "Falta testar a ponte do Norte em 48 kHz.",
                "Na verdade, troquei para o preset Comet porque a CPU do Safira ficava alta.",
                "O teste da ponte do Norte em 48 kHz ficou concluído.",
            ):
                store.add_message(store.load(), text)

            store.select("norte-ii")
            for text in (
                "Comecei o projeto Norte II com o preset Lume no bass.",
                "No Norte II, troquei para o preset Safira porque a CPU do Lume ficava alta.",
                "O teste do Safira no refrão ficou concluído.",
                "Falta testar o refrão do Norte II.",
            ):
                store.add_message(store.load(), text)

            store.select("vale")
            for text in (
                "Comecei o projeto Vale com o preset Atlas.",
                "No Vale, troquei para o preset Comet porque a CPU do Atlas ficava alta.",
                "O teste do Comet no refrão ficou concluído.",
                "Ainda não defini o próximo passo do Vale.",
            ):
                store.add_message(store.load(), text)

            # Reinício lógico: novo store lê o índice e os arquivos persistidos.
            restarted = ProjectStore(directory)
            self.assertEqual(restarted.active_id(), "vale")

            restarted.select("norte")
            north = restarted.resume()
            self.assertEqual(north["decision"]["value"], "Comet")
            self.assertEqual(north["reason"], "a CPU do Safira ficava alta")
            self.assertIn("ponte do Norte", north["completed"]["text"])
            self.assertIsNone(north["pending"])
            self.assertEqual(
                [item["to"] for item in north["corrections"]],
                ["Safira", "Comet"],
            )
            self.assertIn("m1", north["corrections"][0]["source_ids"])
            self.assertIn("m2", north["corrections"][0]["source_ids"])
            text = format_resume(north)
            self.assertIn("Decisão técnica: Comet [m5]", text)
            self.assertIn("Último teste observado", text)

            restarted.select("norte-ii")
            south = restarted.resume()
            self.assertEqual(south["decision"]["value"], "Safira")
            self.assertIn("refrão do Norte II", south["pending"]["text"])
            self.assertNotEqual(south["decision"]["value"], north["decision"]["value"])
            self.assertNotIn("ponte do Norte", (south["completed"] or {}).get("text", ""))

            restarted.select("vale")
            empty = restarted.resume()
            self.assertIsNone(empty["pending"])
            self.assertIsNotNone(empty["no_pending"])
            self.assertIn("sem próximo passo", format_resume(empty))

            restarted.select("norte")
            source = restarted.show("m5")
            self.assertIn("Comet", source["text"])
            self.assertEqual(source["project_id"], "norte")

    def test_initial_reason_preserves_first_letter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sinapse")
            store.add_message(
                store.load(),
                "Comecei a Sinapse com o preset Aurora no bass porque queria um som suave.",
            )
            result = store.resume()
            self.assertEqual(result["reason"], "queria um som suave")
            self.assertTrue(result["reason"].startswith("queria"))

    def test_dashboard_focus_and_checklist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sinapse")
            for text in (
                "Comecei a Sinapse com o preset Aurora no bass.",
                "O teste do refrão ficou concluído.",
                "Falta testar a ponte em 96 kHz.",
            ):
                store.add_message(store.load(), text)
            dashboard = store.dashboard()
            self.assertIsNone(dashboard["goal"])
            self.assertIn("ponte", dashboard["current_task"])
            self.assertEqual(dashboard["focus"]["status"], "proposed")
            self.assertIn("ponte", dashboard["focus"]["action"])
            self.assertEqual(
                [item["status"] for item in dashboard["checklist"]],
                ["done", "pending"],
            )
            self.assertIn("ponte", dashboard["next_step"]["text"])

    def test_show_respects_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sigilo")
            store.add_message(store.load(), "Comecei o projeto Sigilo com o preset X.")
            store.add_message(store.load(), "O teste do segredo ficou concluído.")
            store.restrict("segredo")
            with self.assertRaises(ValueError):
                store.show("m2")

    def test_delete_removes_only_target_and_its_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            norte = store.create("Norte")
            norte_ii = store.create("Norte II")
            store.select("norte")
            store.add_message(store.load(), "Comecei o projeto Norte com o preset Aurora.")
            store.select("norte-ii")
            store.add_message(store.load(), "Comecei o projeto Norte II com o preset Lume.")
            self.assertTrue(store.diagnostics.events("norte"))
            self.assertTrue(store.diagnostics.events("norte-ii"))
            with self.assertRaises(ValueError):
                store.delete("norte-ii", "norte", "Norte II")
            self.assertTrue((Path(directory) / "norte-ii.json").exists())
            result = store.delete("norte-ii", "norte-ii", "Norte II")
            self.assertEqual(result["project_id"], "norte-ii")
            self.assertFalse((Path(directory) / "norte-ii.json").exists())
            self.assertEqual(store.diagnostics.events("norte-ii"), [])
            self.assertTrue(store.diagnostics.events("norte"))
            self.assertEqual(store.active_id(), "norte")

    def test_sensitivity_omits_restricted_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(directory)
            store.create("Sigilo")
            store.add_message(store.load(), "Comecei o projeto Sigilo com o preset X.")
            store.add_message(
                store.load(), "O teste do segredo ficou concluído."
            )
            store.restrict("segredo")
            result = store.resume()
            self.assertIsNone(result["completed"])
            self.assertTrue(result["restricted"])


if __name__ == "__main__":
    unittest.main()
