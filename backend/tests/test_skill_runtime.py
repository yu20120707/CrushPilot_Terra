import json
import tempfile
import unittest
from pathlib import Path

from app.knowledge.skill import SkillCompilationError, SkillRuntime, compile_skill


SKILL_DIR = (
    Path(__file__).parents[1]
    / "app"
    / "agents"
    / "assistant"
    / "resources"
    / "goutoujunshi_skill"
)


class SkillRuntimeTests(unittest.TestCase):
    def test_compile_and_load_produce_review_and_runtime_views(self):
        artifact = compile_skill(SKILL_DIR)
        runtime = SkillRuntime(SKILL_DIR).load()
        view = runtime.view(["explicit_rejection"])

        self.assertEqual(artifact["source_sha256"], runtime.source_hash)
        self.assertEqual(view.version, "2.0.0")
        self.assertEqual(list(view.scene_policies), ["explicit_rejection"])
        self.assertIn("不继续施压", view.prompt)
        self.assertIn("自动生成", (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))

    def test_readiness_rejects_source_changed_after_compilation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in ("skill.yaml", "schema.json"):
                (directory / filename).write_bytes((SKILL_DIR / filename).read_bytes())
            compile_skill(directory)
            (directory / "skill.yaml").write_text(
                (directory / "skill.yaml").read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )
            status = SkillRuntime(directory).readiness()

        self.assertFalse(status.ready)
        self.assertIn("hash", status.errors[0])

    def test_compile_rejects_invalid_version_empty_rules_and_missing_required_scene(self):
        source = (SKILL_DIR / "skill.yaml").read_text(encoding="utf-8")
        cases = (
            source.replace("version: 2.0.0", "version: latest"),
            source.replace(
                "core_rules:\n  - id: fact_inference_separation",
                "core_rules: []\nunused_rules:\n  - id: fact_inference_separation",
            ),
            source.replace("  emotional_support:", "  other_support:"),
        )
        for invalid_source in cases:
            with self.subTest(source=invalid_source[:40]), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "skill.yaml").write_text(invalid_source, encoding="utf-8")
                (directory / "schema.json").write_bytes((SKILL_DIR / "schema.json").read_bytes())
                with self.assertRaises(SkillCompilationError):
                    compile_skill(directory)

    def test_readiness_rejects_missing_or_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in ("skill.yaml", "schema.json"):
                (directory / filename).write_bytes((SKILL_DIR / filename).read_bytes())
            self.assertFalse(SkillRuntime(directory).readiness().ready)
            artifact = compile_skill(directory)
            artifact["skill_version"] = "9.9.9"
            (directory / "skill.compiled.json").write_text(json.dumps(artifact), encoding="utf-8")
            self.assertFalse(SkillRuntime(directory).readiness().ready)

    def test_readiness_rejects_tampered_compiled_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in ("skill.yaml", "schema.json"):
                (directory / filename).write_bytes((SKILL_DIR / filename).read_bytes())
            artifact = compile_skill(directory)
            artifact["config"]["core_rules"][0]["instruction"] = "tampered"
            (directory / "skill.compiled.json").write_text(json.dumps(artifact), encoding="utf-8")

            status = SkillRuntime(directory).readiness()

        self.assertFalse(status.ready)
        self.assertIn("does not match skill.yaml", status.errors[0])


if __name__ == "__main__":
    unittest.main()
