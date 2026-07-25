import unittest
from pathlib import Path

import yaml


class DeployConfigTests(unittest.TestCase):
    def test_corpus_is_published_before_ready_backend_and_frontend(self):
        root = Path(__file__).resolve().parents[2]
        config = yaml.safe_load(
            (root / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
        )
        services = config["services"]
        self.assertEqual(services["postgres"]["image"], "pgvector/pgvector:pg16")
        self.assertEqual(
            services["backend"]["depends_on"]["corpus-publisher"]["condition"],
            "service_completed_successfully",
        )
        self.assertIn("/ready", " ".join(services["backend"]["healthcheck"]["test"]))
        self.assertEqual(
            services["frontend"]["depends_on"]["backend"]["condition"],
            "service_healthy",
        )
        self.assertIn("publish_corpus.py", " ".join(services["corpus-publisher"]["command"]))


if __name__ == "__main__":
    unittest.main()
