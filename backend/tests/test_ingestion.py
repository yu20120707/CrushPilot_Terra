import json
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml

from app.knowledge.ingestion import (
    ARTIFACT_FILENAMES,
    build_artifact_paths,
    build_corpus,
    chunk_sections,
    parse_markdown,
    token_count,
    write_build_artifacts,
)


class IngestionTests(unittest.TestCase):
    def test_heading_aware_parser_ignores_headings_in_fences(self):
        sections = parse_markdown(
            "# Root\n\nintro\n\n## Child\n\nbody\n```\n# not heading\n```",
            "fallback",
        )
        self.assertEqual(sections[0].heading_path, ("Root",))
        self.assertEqual(sections[1].heading_path, ("Root", "Child"))
        self.assertIn("# not heading", sections[1].content)
        self.assertEqual(sections[1].blocks[-1].kind, "code")

    def test_setext_heading_and_block_ast(self):
        sections = parse_markdown(
            "Root\n====\n\nintro\n\nChild\n-----\n\n- one\n- two",
            "fallback",
        )
        self.assertEqual(sections[0].heading_path, ("Root",))
        self.assertEqual(sections[0].blocks[0].kind, "paragraph")
        self.assertEqual(sections[1].heading_path, ("Root", "Child"))
        self.assertEqual(sections[1].blocks[0].kind, "list")

    def test_fence_requires_matching_marker_and_sufficient_length(self):
        sections = parse_markdown(
            "````python\n# not a heading\n~~~\n### still code\n```\n## also code\n````\n# Real #\n\nbody",
            "fallback",
        )
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].heading_path, ("fallback",))
        self.assertIn("### still code", sections[0].content)
        self.assertEqual(sections[1].heading_path, ("Real",))
        self.assertEqual(sections[1].content, "body")

    def test_long_content_is_bounded_and_overlapped(self):
        sections = parse_markdown("# Long\n\n" + " ".join(map(str, range(1400))), "x")
        chunks = chunk_sections(sections)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.token_count <= 900 for chunk in chunks))
        first_tokens = chunks[0].content.split()
        second_tokens = chunks[1].content.split()
        self.assertEqual(first_tokens[-80:], second_tokens[:80])

    def test_builds_all_required_sources_with_deterministic_metadata(self):
        first = build_corpus("test-v1")
        second = build_corpus("test-v1")
        self.assertEqual(first.report.document_count, 40)
        self.assertEqual(
            [item.chunk.chunk_id for item in first.chunks],
            [item.chunk.chunk_id for item in second.chunks],
        )
        next_version = build_corpus("test-v2")
        self.assertNotEqual(
            [item.chunk.document_id for item in first.chunks],
            [item.chunk.document_id for item in next_version.chunks],
        )
        self.assertNotEqual(
            [item.chunk.chunk_id for item in first.chunks],
            [item.chunk.chunk_id for item in next_version.chunks],
        )
        self.assertTrue(all(item.chunk.token_count > 0 for item in first.chunks))
        self.assertTrue(all(source["chunk_count"] > 0 for source in first.manifest["sources"]))
        self.assertLess(first.report.missing_metadata_ratios["topics"], 1.0)
        self.assertEqual(first.report.missing_metadata_ratios["knowledge_type"], 0.0)
        self.assertTrue(
            any(
                "boundary" in item.chunk.topics
                and "stop" in item.chunk.action_labels
                for item in first.chunks
            )
        )
        self.assertTrue(
            all(item.chunk.review_status == "approved" for item in first.chunks)
        )
        self.assertTrue(
            all(
                uuid.UUID(value)
                for item in first.chunks
                for value in (
                    item.chunk.document_id,
                    item.chunk.chunk_id,
                    *(
                        [item.chunk.parent_section_id]
                        if item.chunk.parent_section_id
                        else []
                    ),
                )
            )
        )
        documents = {}
        for item in first.chunks:
            documents.setdefault(item.chunk.document_id, []).append(item)
        for items in documents.values():
            self.assertIsNone(items[0].relationships.previous_chunk_id)
            self.assertIsNone(items[-1].relationships.next_chunk_id)
            for index, item in enumerate(items):
                self.assertEqual(
                    item.relationships.previous_chunk_id,
                    items[index - 1].chunk.chunk_id if index else None,
                )
                self.assertEqual(
                    item.relationships.next_chunk_id,
                    items[index + 1].chunk.chunk_id
                    if index + 1 < len(items)
                    else None,
                )

    def test_artifact_writer_round_trip(self):
        build = build_corpus("test-v1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(
                build_artifact_paths(output),
                {
                    name: output / filename
                    for name, filename in ARTIFACT_FILENAMES.items()
                },
            )
            artifacts = write_build_artifacts(build, output)
            self.assertEqual(
                artifacts["manifest"].name, "source-manifest.yaml"
            )
            manifest = yaml.safe_load(
                artifacts["manifest"].read_text(encoding="utf-8")
            )
            counts = json.loads(
                artifacts["chunk_counts"].read_text(encoding="utf-8")
            )
            report = json.loads(
                artifacts["build_report"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest, build.manifest)
            self.assertEqual(sum(counts.values()), len(build.chunks))
            self.assertEqual(report["chunk_count"], len(build.chunks))
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_required_empty_document_fails_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "practical").mkdir()
            (root / "knowledge" / "empty.md").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero chunks"):
                build_corpus("test", root)

    def test_token_count_handles_chinese_and_words(self):
        self.assertEqual(token_count("你好 hello world"), 4)


if __name__ == "__main__":
    unittest.main()
