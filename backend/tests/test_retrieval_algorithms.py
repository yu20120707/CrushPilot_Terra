import unittest

from app.knowledge.retrieval.diversity import select_diverse
from app.knowledge.retrieval.evidence_gate import assess_evidence
from app.knowledge.retrieval.fusion import weighted_rrf
from app.knowledge.retrieval.reranker import RerankerUnavailable, rerank


def candidate(chunk_id, document_id="doc", **values):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "topics": [],
        "action_labels": [],
        "knowledge_type": "principle",
        "content": chunk_id,
        **values,
    }


class RetrievalAlgorithmTests(unittest.TestCase):
    def test_weighted_rrf_uses_rank_and_query_weights(self):
        scores = weighted_rrf(
            {("lexical", 0): 1, ("both", 0): 2},
            {("vector", 0): 1, ("both", 0): 2},
            [1.0],
        )
        self.assertGreater(scores["both"], scores["vector"])
        self.assertGreater(scores["vector"], scores["lexical"])

    def test_diversity_filters_excluded_and_caps_documents(self):
        selected = select_diverse(
            [
                candidate("bad", topics=["aggressive"]),
                candidate("a", topics=["boundary"], rerank_score=0.9),
                candidate("b", topics=["boundary"], rerank_score=0.8),
                candidate("c", topics=["boundary"], rerank_score=0.7),
                candidate("d", document_id="other", topics=["rejection"]),
            ],
            ["boundary", "rejection"],
            ["aggressive"],
        )
        self.assertEqual([item["chunk_id"] for item in selected], ["a", "d", "b"])

    def test_diversity_covers_distinct_required_topics_and_drops_overlap(self):
        selected = select_diverse(
            [
                candidate(
                    "a",
                    topics=["boundary"],
                    content="明确边界 后 不再 继续 施压",
                    rerank_score=0.9,
                ),
                candidate(
                    "duplicate",
                    document_id="other",
                    topics=["boundary"],
                    content="明确边界后不再继续施压",
                    rerank_score=0.8,
                ),
                candidate(
                    "rejection",
                    document_id="third",
                    topics=["rejection"],
                    rerank_score=0.7,
                ),
            ],
            ["boundary", "rejection"],
            [],
        )
        self.assertEqual(
            {item["chunk_id"] for item in selected}, {"a", "rejection"}
        )

    def test_evidence_gate_all_states(self):
        self.assertEqual(assess_evidence([], ["boundary"], []).status, "insufficient")
        self.assertEqual(
            assess_evidence(
                [candidate("a", topics=["boundary"])],
                ["boundary", "rejection"],
                [],
            ).status,
            "partially_sufficient",
        )
        self.assertEqual(
            assess_evidence(
                [
                    candidate("a", topics=["boundary"], action_labels=["advance"]),
                    candidate("b", topics=["boundary"], action_labels=["stop"]),
                ],
                ["boundary"],
                [],
            ).status,
            "conflicting",
        )
        self.assertEqual(
            assess_evidence(
                [candidate("a", topics=["boundary"])],
                ["boundary"],
                [],
            ).status,
            "sufficient",
        )

    def test_reranker_enforces_score_shape_and_order(self):
        class Good:
            model_name = "test"

            def score(self, scene, query, documents):
                return [0.1, 0.9]

        self.assertEqual(
            rerank([candidate("a"), candidate("b")], "scene", "query", Good())[0][
                "chunk_id"
            ],
            "b",
        )

        class Bad(Good):
            def score(self, scene, query, documents):
                return []

        with self.assertRaises(RerankerUnavailable):
            rerank([candidate("a")], "scene", "query", Bad())


if __name__ == "__main__":
    unittest.main()
