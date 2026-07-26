import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.detect_legacy_retrieval_paths import detect
from scripts.verify_corpus_version import verify as verify_corpus_version
from scripts.verify_traceability import verify as verify_traceability


class GovernanceScriptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def _write_yaml(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def test_traceability_passes_only_completed_tasks_with_existing_evidence(self):
        evidence = self.root / "evidence.txt"
        evidence.write_text("verified", encoding="utf-8")
        trace = self.root / "traceability.yaml"
        tasks = self.root / "tasks"
        self._write_yaml(
            trace,
            {
                "schema_version": 2,
                "requirements": [
                    {
                        "id": "SPEC-001",
                        "source_sections": ["design/1"],
                        "summary": "requirement",
                        "tasks": ["01-task"],
                        "evidence": {"files": ["evidence.txt"]},
                    }
                ]
            },
        )
        self._write_yaml(
            tasks / "01-task/status.yaml",
            {
                "task": "01-task",
                "spec_ids": ["SPEC-001"],
                "status": "completed",
                "evidence": ["evidence.txt"],
            },
        )
        result = verify_traceability(trace, tasks, self.root, {"SPEC-001"})
        self.assertEqual(result["unmapped_requirements"], 0)
        self.assertEqual(result["unverified_tasks"], 0)

    def test_traceability_counts_unmapped_and_pending_without_crashing(self):
        trace = self.root / "traceability.yaml"
        tasks = self.root / "tasks"
        self._write_yaml(
            trace,
            {
                "schema_version": 2,
                "requirements": [
                    {
                        "id": "SPEC-001",
                        "source_sections": ["design/1"],
                        "summary": "requirement",
                        "tasks": ["01-task"],
                        "evidence": {"files": ["missing.txt"]},
                    }
                ]
            },
        )
        self._write_yaml(
            tasks / "01-task/status.yaml",
            {
                "task": "01-task",
                "spec_ids": ["SPEC-001", "SPEC-002"],
                "status": "pending",
            },
        )
        result = verify_traceability(
            trace, tasks, self.root, {"SPEC-001", "SPEC-002"}
        )
        self.assertEqual(result["unmapped_requirements"], 2)
        self.assertEqual(result["unverified_tasks"], 1)

    def test_traceability_requires_exact_bidirectional_task_mapping(self):
        evidence = self.root / "evidence.txt"
        evidence.write_text("verified", encoding="utf-8")
        trace = self.root / "traceability.yaml"
        tasks = self.root / "tasks"
        self._write_yaml(
            trace,
            {
                "schema_version": 2,
                "requirements": [
                    {
                        "id": "SPEC-001",
                        "source_sections": ["design/1"],
                        "summary": "requirement",
                        "tasks": ["01-task"],
                        "evidence": {"files": ["evidence.txt"]},
                    }
                ]
            },
        )
        self._write_yaml(
            tasks / "01-task/status.yaml",
            {
                "task": "01-task",
                "spec_ids": ["SPEC-001", "SPEC-EXTRA"],
                "status": "completed",
                "evidence": ["evidence.txt"],
            },
        )

        result = verify_traceability(trace, tasks, self.root, {"SPEC-001"})

        self.assertEqual(result["unmapped_requirements"], 1)
        self.assertEqual(result["unverified_tasks"], 1)

    def test_traceability_rejects_wrong_schema_and_missing_required_task(self):
        trace = self.root / "traceability.yaml"
        tasks = self.root / "tasks"
        self._write_yaml(trace, {"schema_version": 1, "requirements": []})

        result = verify_traceability(
            trace,
            tasks,
            self.root,
            set(),
            {"01-task"},
        )

        self.assertEqual(result["unmapped_requirements"], 1)
        self.assertEqual(result["details"]["missing_tasks"], ["01-task"])
        self.assertEqual(result["unverified_tasks"], 1)

    def test_traceability_rejects_string_tasks_and_duplicate_task_specs(self):
        evidence = self.root / "evidence.txt"
        evidence.write_text("verified", encoding="utf-8")
        trace = self.root / "traceability.yaml"
        tasks = self.root / "tasks"
        self._write_yaml(
            trace,
            {
                "schema_version": 2,
                "requirements": [
                    {
                        "id": "SPEC-001",
                        "source_sections": ["design/1"],
                        "summary": "requirement",
                        "tasks": "01-task",
                        "evidence": {"files": ["evidence.txt"]},
                    }
                ],
            },
        )
        self._write_yaml(
            tasks / "01-task/status.yaml",
            {
                "task": "01-task",
                "spec_ids": ["SPEC-001", "SPEC-001"],
                "status": "completed",
                "evidence": ["evidence.txt"],
            },
        )

        result = verify_traceability(
            trace,
            tasks,
            self.root,
            {"SPEC-001"},
            {"01-task"},
        )

        self.assertGreater(result["unmapped_requirements"], 0)
        self.assertEqual(result["details"]["invalid_tasks"], ["01-task"])

    def test_corpus_version_requires_matching_versions_models_and_counts(self):
        manifest = self.root / "manifest.yaml"
        report = self.root / "build-report.json"
        expected = self.root / "expected.json"
        self._write_yaml(
            manifest,
            {
                "corpus_version": "2026.07.1",
                "sources": [
                    {"path": "one.md", "status": "ingested", "chunk_count": 3}
                ],
            },
        )
        expected_value = {
            "corpus_version": "2026.07.1",
            "embedding_model": "bge-m3",
            "tokenizer_version": "jieba-0.42",
            "chunker_version": "semantic-v1",
        }
        report_value = {
            **expected_value,
            "document_count": 1,
            "chunk_count": 3,
        }
        report.write_text(json.dumps(report_value), encoding="utf-8")
        expected.write_text(json.dumps(expected_value), encoding="utf-8")
        self.assertEqual(
            verify_corpus_version(manifest, report, expected)[
                "corpus_version_mismatches"
            ],
            0,
        )
        report_value["tokenizer_version"] = "wrong"
        report.write_text(json.dumps(report_value), encoding="utf-8")
        self.assertGreater(
            verify_corpus_version(manifest, report, expected)[
                "corpus_version_mismatches"
            ],
            0,
        )

    def test_legacy_detector_clean_and_each_forbidden_family(self):
        clean = self.root / "clean"
        clean.mkdir()
        (clean / "service.py").write_text(
            "# knowledge_vectors.json was removed\n"
            "def retrieve(query):\n    return repository.search(query)\n",
            encoding="utf-8",
        )
        self.assertEqual(detect(clean)["legacy_retrieval_paths"], 0)

        legacy = self.root / "legacy"
        legacy.mkdir()
        (legacy / "old.py").write_text(
            "\n".join(
                [
                    "selected_reference_paths = {}",
                    "index = 'knowledge_vectors.json'",
                    "score = text.count(term)",
                    "prompt = '<GOUTOUJUNSHI_REFERENCES>'",
                    "CHAT_SYSTEM_PROMPT = RUNTIME_INSTRUCTIONS",
                ]
            ),
            encoding="utf-8",
        )
        result = detect(legacy)
        categories = {
            finding["category"] for finding in result["details"]["findings"]
        }
        self.assertEqual(categories, set(detect.__globals__["PATTERNS"]))
        self.assertGreaterEqual(result["legacy_retrieval_paths"], 5)


if __name__ == "__main__":
    unittest.main()
