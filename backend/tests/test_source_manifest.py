import tempfile
import unittest
from pathlib import Path

from app.knowledge.governance.source_manifest import (
    build_manifest,
    read_manifest,
    scan_sources,
    validate_manifest,
    write_manifest,
)


class SourceManifestTests(unittest.TestCase):
    def test_manifest_covers_only_required_reference_groups(self):
        paths = scan_sources()
        counts = {
            path.relative_to(path.parents[1]).as_posix(): 1
            for path in paths
        }
        manifest = build_manifest("test", chunk_counts=counts)
        self.assertEqual(
            {source["path"] for source in manifest["sources"]},
            set(counts),
        )
        self.assertFalse(validate_manifest(manifest))
        self.assertTrue(
            all(
                source["path"].startswith(("knowledge/", "practical/"))
                for source in manifest["sources"]
            )
        )

    def test_required_zero_chunk_or_changed_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            source = root / "knowledge" / "one.md"
            source.write_text("# One\n\nContent", encoding="utf-8")
            manifest = build_manifest("test", root=root)
            errors = validate_manifest(manifest, root)
            self.assertTrue(any("invalid release status" in error for error in errors))
            self.assertTrue(any("required source has zero chunks" in error for error in errors))
            manifest = build_manifest(
                "test",
                chunk_counts={"knowledge/one.md": 1},
                root=root,
            )
            source.write_text("changed", encoding="utf-8")
            self.assertIn("source hash mismatch", validate_manifest(manifest, root)[0])

    def test_yaml_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            (root / "knowledge" / "one.md").write_text("content", encoding="utf-8")
            manifest = build_manifest(
                "test",
                chunk_counts={"knowledge/one.md": 1},
                root=root,
            )
            output = root / "manifest.yaml"
            write_manifest(manifest, output)
            self.assertEqual(read_manifest(output), manifest)


if __name__ == "__main__":
    unittest.main()
