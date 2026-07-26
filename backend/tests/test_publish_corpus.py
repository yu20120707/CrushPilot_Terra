import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.knowledge.ingestion import build_corpus
from app.knowledge.governance.corpus_version import CURRENT_CORPUS_VERSION
from scripts.publish_corpus import embedding_text, search_tokens


class PublishCorpusTests(unittest.TestCase):
    def test_embedding_recipe_uses_a_new_corpus_identity(self):
        self.assertEqual(CURRENT_CORPUS_VERSION, "2026.07.6")
        version_artifact = (
            ROOT / "trellis/retrieval-system-refactor/corpus-version.json"
        )
        self.assertEqual(
            __import__("json").loads(version_artifact.read_text(encoding="utf-8"))[
                "corpus_version"
            ],
            CURRENT_CORPUS_VERSION,
        )

    def test_search_tokens_cover_every_chunk_and_four_weights(self):
        build = build_corpus("test-publish")
        tokens = search_tokens(build)
        self.assertEqual(
            set(tokens), {record.chunk.chunk_id for record in build.chunks}
        )
        self.assertTrue(all(set(fields) == {"A", "B", "C", "D"} for fields in tokens.values()))
        boundary = next(
            record.chunk
            for record in build.chunks
            if "boundary" in record.chunk.topics
        )
        self.assertIn("boundary", tokens[boundary.chunk_id]["A"])
        text = embedding_text(boundary)
        self.assertIn(boundary.title, text)
        self.assertIn("边界", text)
        self.assertIn(boundary.content, text)


if __name__ == "__main__":
    unittest.main()
