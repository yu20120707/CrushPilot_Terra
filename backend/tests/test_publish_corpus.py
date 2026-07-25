import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.knowledge.ingestion import build_corpus
from scripts.publish_corpus import search_tokens


class PublishCorpusTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
