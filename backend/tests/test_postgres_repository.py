from __future__ import annotations

import json
import os
import unittest
import uuid
from types import SimpleNamespace

from app.db.schema import SCHEMA_STATEMENTS, apply_schema
from app.knowledge.governance.corpus_version import CorpusVersion
from app.knowledge.repositories.postgres import PostgresKnowledgeRepository, _safe_reason
from app.knowledge.repositories.publisher import CorpusPublisher


class FakeCursor:
    def __init__(self, connection: "FakeConnection"):
        self.connection = connection
        self.description = [
            SimpleNamespace(name=name) for name in connection.columns
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute(self, sql, params=None):
        if self.connection.fail_on_execute:
            raise RuntimeError("database write failed")
        self.connection.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.connection.rows.pop(0) if self.connection.rows else None

    def fetchall(self):
        rows, self.connection.rows = self.connection.rows, []
        return rows


class FakeConnection:
    def __init__(self, rows=None, columns=None, fail_on_execute=False):
        self.rows = list(rows or [])
        self.columns = list(columns or [])
        self.fail_on_execute = fail_on_execute
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def corpus(version: str, model: str = "embed-v1") -> CorpusVersion:
    return CorpusVersion(version, model, 3, "l2", "tokens-v1", "chunks-v1")


class SchemaAndRepositoryTests(unittest.TestCase):
    def test_compound_fallback_reason_is_preserved_but_arbitrary_text_is_rejected(self):
        self.assertEqual(
            _safe_reason("vector_unavailable,reranker_unavailable"),
            "vector_unavailable,reranker_unavailable",
        )
        self.assertIsNone(_safe_reason("private detail"))

    def test_generation_failure_updates_the_requested_trace_id(self):
        connection = FakeConnection()
        repository = PostgresKnowledgeRepository(connection)
        repository.mark_trace_failure("trace-123", "generation_failed", 12.5)
        sql, params = connection.executed[-1]
        self.assertIn("UPDATE retrieval_traces", sql)
        self.assertEqual(params, ("generation_failed", 12.5, "trace-123"))

    def test_schema_has_exactly_seven_tables_and_no_approximate_vector_index(self):
        sql = "\n".join(SCHEMA_STATEMENTS).lower()
        self.assertEqual(sql.count("create table if not exists"), 7)
        self.assertIn("create extension if not exists vector", sql)
        self.assertNotIn("hnsw", sql)
        self.assertNotIn("ivfflat", sql)
        self.assertIn("using gin(search_vector)", sql)
        self.assertIn("using gin(metadata)", sql)
        self.assertIn("unique(id, corpus_version)", sql)
        self.assertIn("foreign key(chunk_id, corpus_version)", sql)
        self.assertIn("alter table knowledge_chunks", sql)
        self.assertIn("alter table knowledge_embeddings", sql)
        self.assertIn("if not exists", sql)

    def test_vector_search_uses_exact_cosine_and_published_version(self):
        connection = FakeConnection()
        repository = PostgresKnowledgeRepository(connection)
        repository.exact_cosine_search([0.1, 0.2, 0.3], "embed-v1")
        sql, params = connection.executed[0]
        self.assertIn("e.embedding <=> %s::vector", sql)
        self.assertIn("v.status = 'published'", sql)
        self.assertIn("e.corpus_version = c.corpus_version", sql)
        self.assertNotIn("hnsw", sql.lower())
        self.assertEqual(
            params,
            (
                "[0.1,0.2,0.3]", "embed-v1", 3,
                None, None, None, None, None, None, None, None,
                "[0.1,0.2,0.3]", 20,
            ),
        )

    def test_vector_search_hard_filters_are_fixed_and_parameterized(self):
        connection = FakeConnection()
        filters = {
            "review_status": ["review-sentinel"],
            "task_types": ["reply"],
            "relationship_stages": ["early"],
            "knowledge_type": ["strategy"],
        }
        PostgresKnowledgeRepository(connection).exact_cosine_search(
            [1, 0, 0], "embed-v1", hard_filters=filters
        )
        sql, params = connection.executed[0]
        self.assertIn("c.review_status = ANY(%s::text[])", sql)
        self.assertIn("c.metadata -> 'task_types' ?| %s::text[]", sql)
        self.assertIn("c.metadata -> 'relationship_stages' ?| %s::text[]", sql)
        self.assertIn("c.metadata ->> 'knowledge_type' = ANY(%s::text[])", sql)
        self.assertNotIn("review-sentinel", sql)
        self.assertEqual(
            params[3:11],
            (
                ["review-sentinel"], ["review-sentinel"], ["reply"], ["reply"],
                ["early"], ["early"], ["strategy"], ["strategy"],
            ),
        )

    def test_vector_search_rejects_unknown_and_empty_hard_filters(self):
        repository = PostgresKnowledgeRepository(FakeConnection())
        with self.assertRaises(ValueError):
            repository.exact_cosine_search(
                [1, 0, 0], "embed-v1", hard_filters={"private_id": ["x"]}
            )
        with self.assertRaises(ValueError):
            repository.exact_cosine_search(
                [1, 0, 0], "embed-v1", hard_filters={"task_types": []}
            )

    def test_vector_search_returns_normalizable_candidate_shape(self):
        columns = [
            "chunk_id", "document_id", "parent_section_id", "title",
            "heading_path", "content", "metadata", "score",
        ]
        row = (
            "chunk-1", "doc-1", "section-1", "Title", ["Root"],
            "Content", {"token_count": 12, "previous_chunk_id": "chunk-0"}, 0.95,
        )
        result = PostgresKnowledgeRepository(
            FakeConnection([row], columns)
        ).exact_cosine_search([0.1, 0.2, 0.3], "embed-v1")
        self.assertEqual(set(result[0]), set(columns))
        self.assertEqual(result[0]["chunk_id"], "chunk-1")
        self.assertEqual(result[0]["metadata"]["token_count"], 12)
        self.assertEqual(result[0]["metadata"]["previous_chunk_id"], "chunk-0")

    def test_staging_build_persists_weighted_search_and_complete_metadata(self):
        chunk = SimpleNamespace(
            chunk_id="chunk-1", document_id="doc-1", parent_section_id="section-1",
            title="Title", heading_path=["Root", "Title"], content="Body",
            topics=["topic"], task_types=["reply"], relationship_stages=["early"],
            action_labels=["act"], applicable_when=["yes"], not_applicable_when=["no"],
            knowledge_type="source_note", evidence_level="L4", review_status="approved",
            priority=2, source_path="knowledge/one.md", source_sha256="sha",
            corpus_version="v1", token_count=12,
        )
        build = SimpleNamespace(
            report=SimpleNamespace(
                corpus_version="v1", embedding_model="embed-v1",
                embedding_dimension=3, embedding_normalization="l2",
                tokenizer_version="tokens-v1", chunker_version="chunks-v1",
            ),
            chunks=[SimpleNamespace(
                chunk=chunk,
                relationships=SimpleNamespace(
                    previous_chunk_id="chunk-0", next_chunk_id="chunk-2",
                ),
            )],
        )
        connection = FakeConnection()
        PostgresKnowledgeRepository(connection).write_staging_build(
            build, {"chunk-1": {
                "A": "title topic act", "B": "root title yes no",
                "C": "body", "D": "knowledge one md",
            }}
        )
        chunk_sql, params = next(
            item for item in connection.executed
            if "INSERT INTO knowledge_chunks" in item[0]
        )
        self.assertEqual(chunk_sql.count("setweight(to_tsvector('simple'"), 4)
        self.assertIn("'A'", chunk_sql)
        self.assertIn("'B'", chunk_sql)
        self.assertIn("'C'", chunk_sql)
        self.assertIn("'D'", chunk_sql)
        metadata = json.loads(params[6])
        self.assertEqual(metadata["token_count"], 12)
        self.assertEqual(metadata["previous_chunk_id"], "chunk-0")
        self.assertEqual(metadata["next_chunk_id"], "chunk-2")
        self.assertEqual(
            params[7],
            "title topic act root title yes no body knowledge one md",
        )

    def test_embedding_batch_validates_model_and_dimension(self):
        mismatch = FakeConnection([("embed-v1", 3)])
        with self.assertRaises(ValueError):
            PostgresKnowledgeRepository(mismatch).write_embeddings(
                "v1", "wrong-model", {"chunk-1": [1, 0, 0]}
            )
        self.assertEqual(mismatch.rollbacks, 1)

        invalid = FakeConnection([("embed-v1", 3)])
        with self.assertRaises(ValueError):
            PostgresKnowledgeRepository(invalid).write_embeddings(
                "v1", "embed-v1", {"chunk-1": [1, 0]}
            )
        self.assertEqual(invalid.rollbacks, 1)

    def test_readiness_enforces_corpus_and_embedding_metadata(self):
        expected = corpus("v1")
        matching = FakeConnection([("v1", "embed-v1", 3, "l2", "tokens-v1", "chunks-v1", 2, 2)])
        self.assertTrue(PostgresKnowledgeRepository(matching).readiness(expected).ready)

        mismatch = FakeConnection([("v1", "embed-v2", 3, "l2", "tokens-v1", "chunks-v1", 2, 2)])
        status = PostgresKnowledgeRepository(mismatch).readiness(expected)
        self.assertFalse(status.ready)
        degraded = FakeConnection([("v1", "embed-v2", 3, "l2", "tokens-v1", "chunks-v1", 2, 2)])
        status = PostgresKnowledgeRepository(degraded).readiness(expected, allow_lexical_only=True)
        self.assertTrue(status.ready)
        self.assertTrue(status.lexical_only)

        incomplete = FakeConnection([("v1", "embed-v1", 3, "l2", "tokens-v1", "chunks-v1", 2, 1)])
        status = PostgresKnowledgeRepository(incomplete).readiness(expected)
        self.assertFalse(status.ready)
        self.assertEqual(status.reason, "published corpus embeddings are incomplete")
        degraded = FakeConnection([("v1", "embed-v1", 3, "l2", "tokens-v1", "chunks-v1", 2, 1)])
        status = PostgresKnowledgeRepository(degraded).readiness(expected, allow_lexical_only=True)
        self.assertTrue(status.ready)
        self.assertTrue(status.lexical_only)
        self.assertEqual(status.reason, "published corpus embeddings are incomplete")

    def test_publish_is_transactional_and_prunes_to_active_plus_previous(self):
        connection = FakeConnection([(2, 2)])
        CorpusPublisher(connection).publish("v2")
        sql = "\n".join(statement for statement, _ in connection.executed)
        self.assertIn("LOCK TABLE knowledge_corpus_versions IN EXCLUSIVE MODE", sql)
        self.assertIn("status = 'retired'", sql)
        self.assertIn("status = 'published'", sql)
        self.assertIn("LIMIT 1", sql)
        self.assertEqual(connection.commits, 1)

    def test_failed_publish_rolls_back(self):
        connection = FakeConnection([(0, 0)])
        with self.assertRaises(ValueError):
            CorpusPublisher(connection).publish("empty")
        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)

    def test_rollback_revalidates_retained_corpus(self):
        connection = FakeConnection([(1, 0)])
        with self.assertRaises(ValueError):
            CorpusPublisher(connection).rollback("v1")
        sql = "\n".join(statement for statement, _ in connection.executed)
        self.assertIn("status = %s", sql)
        self.assertIn("c.corpus_version = e.corpus_version", sql)
        self.assertEqual(connection.rollbacks, 1)

    def test_trace_boundary_removes_raw_text_deeply(self):
        connection = FakeConnection()
        trace = {
            "request_id": "req-1",
            "conversation_id": "conv-1",
            "skill_version": "skill-v1",
            "skill_sha256": "sha",
            "corpus_version": "v1",
            "embedding_model": "embed-v1",
            "reranker_model": None,
            "scene_snapshot": {
                "current_event_id": "a" * 64,
                "user_goal_id": "b" * 64,
                "known_facts": ["private raw fact"],
                "message_ids": ["m1"],
                "private_id": "suffix-bypass",
                "chunk_id": "invalid id with spaces",
                "task_type_id": "reply",
                "relationship_stage_id": "ambiguous",
                "active_skill_scenario_ids": ["boundary"],
                "confidence": 0.8,
            },
            "retrieval_plan": {
                "query_ids": ["c" * 64],
                "required_topic_ids": ["consent"],
                "excluded_topic_ids": ["coercion"],
                "hard_filter_ids": {"review_status": ["approved"]},
                "soft_preference_ids": {"knowledge_type": ["procedure"]},
            },
            "evidence_assessment": {"status": "sufficient", "selected_chunk_ids": ["c1"]},
            "fallback_reason": "raw reason with spaces",
            "latency_ms": {
                "retrieval": 1.0,
                "scene_analysis_latency_ms": 2.5,
                "generation_latency_ms": "private text latency",
                "rerank_latency_ms": float("inf"),
            },
            "fused_candidates": [
                {"chunk_id": "c1", "rrf_score": 0.8, "content": "private fused content"}
            ],
            "reranked_candidates": [
                {"chunk_id": "c1", "rerank_score": 0.85, "content": "private rerank content"}
            ],
            "selected_chunks": [
                {"chunk_id": "c1", "score": 0.9, "content": "private chunk content"}
            ],
        }
        PostgresKnowledgeRepository(connection).write_trace(trace)
        trace_params = connection.executed[0][1]
        serialized = " ".join(
            value for value in trace_params if isinstance(value, str)
        )
        self.assertNotIn("private raw", serialized)
        self.assertNotIn("suffix-bypass", serialized)
        self.assertNotIn("private text latency", serialized)
        self.assertIsNone(trace_params[10])
        self.assertEqual(
            json.loads(trace_params[7]),
            {
                "task_type_id": "reply",
                "relationship_stage_id": "ambiguous",
                "current_event_id": "a" * 64,
                "user_goal_id": "b" * 64,
                "active_skill_scenario_ids": ["boundary"],
            },
        )
        plan = json.loads(trace_params[8])
        self.assertEqual(plan["required_topic_ids"], ["consent"])
        self.assertEqual(len(plan["query_ids"][0]), 64)
        self.assertEqual(plan["soft_preference_ids"], {"knowledge_type": ["procedure"]})
        self.assertEqual(
            json.loads(trace_params[11]),
            {"scene_analysis_latency_ms": 2.5},
        )
        candidate_params = connection.executed[-1][1]
        self.assertEqual(
            json.loads(candidate_params[-1]),
            {"chunk_id": "c1", "score": 0.9},
        )
        fused_params = connection.executed[1][1]
        reranked_params = connection.executed[2][1]
        self.assertEqual(fused_params[4], 0.8)
        self.assertEqual(reranked_params[4], 0.85)

    def test_trace_write_failure_rolls_back_and_reraises(self):
        trace = {
            "request_id": "req-1", "conversation_id": "conv-1",
            "skill_version": "v1", "skill_sha256": "abc", "corpus_version": "v1",
            "scene_snapshot": {}, "retrieval_plan": {},
            "evidence_assessment": {}, "latency_ms": {},
        }
        connection = FakeConnection(fail_on_execute=True)
        with self.assertRaises(RuntimeError):
            PostgresKnowledgeRepository(connection).write_trace(trace)
        self.assertEqual(connection.rollbacks, 1)

    def test_trace_boundary_rejects_invalid_top_level_identifier(self):
        trace = {
            "request_id": "private id with spaces",
            "conversation_id": "conv-1",
            "skill_version": "skill-v1",
            "skill_sha256": "abc",
            "corpus_version": "v1",
            "scene_snapshot": {},
            "retrieval_plan": {},
            "evidence_assessment": {},
            "latency_ms": {},
        }
        with self.assertRaises(ValueError):
            PostgresKnowledgeRepository(FakeConnection()).write_trace(trace)


@unittest.skipUnless(
    os.getenv("TEST_DATABASE_URL"),
    "TEST_DATABASE_URL is not configured; real PostgreSQL + pgvector integration skipped",
)
class PostgreSQLIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg

        cls.connection = psycopg.connect(os.environ["TEST_DATABASE_URL"])
        apply_schema(cls.connection)

    @classmethod
    def tearDownClass(cls):
        cls._truncate()
        cls.connection.close()

    @classmethod
    def _truncate(cls):
        with cls.connection.cursor() as cursor:
            cursor.execute(
                """
                TRUNCATE retrieval_trace_candidates, retrieval_traces,
                         knowledge_embeddings, knowledge_chunks, knowledge_sections,
                         knowledge_documents, knowledge_corpus_versions CASCADE
                """
            )
        cls.connection.commit()

    def setUp(self):
        self._truncate()

    def _stage(self, version: str, vector: str) -> str:
        repository = PostgresKnowledgeRepository(self.connection)
        repository.create_staging_version(corpus(version))
        document_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_documents
                    (id, source_path, title, source_sha256, corpus_version)
                VALUES (%s, %s, 'Doc', 'sha', %s)
                """,
                (document_id, f"{version}.md", version),
            )
            cursor.execute(
                """
                INSERT INTO knowledge_chunks (
                    id, document_id, title, heading_path, content, metadata,
                    search_tokens, search_vector, review_status, source_sha256,
                    corpus_version
                ) VALUES (%s, %s, 'Chunk', '[]', 'content', '{}', 'content',
                          to_tsvector('simple', 'content'), 'approved', 'sha', %s)
                """,
                (chunk_id, document_id, version),
            )
            cursor.execute(
                """
                INSERT INTO knowledge_embeddings
                    (chunk_id, embedding_model, embedding_dimension, embedding, corpus_version)
                VALUES (%s, 'embed-v1', 3, %s::vector, %s)
                """,
                (chunk_id, vector, version),
            )
        self.connection.commit()
        return chunk_id

    def test_publish_exact_search_readiness_retention_and_rollback(self):
        publisher = CorpusPublisher(self.connection)
        first = self._stage("integration-v1", "[1,0,0]")
        publisher.publish("integration-v1")
        self._stage("integration-v2", "[0,1,0]")
        publisher.publish("integration-v2")
        self._stage("integration-v3", "[0,0,1]")
        publisher.publish("integration-v3")

        with self.connection.cursor() as cursor:
            cursor.execute("SELECT version, status FROM knowledge_corpus_versions ORDER BY version")
            self.assertEqual(
                cursor.fetchall(),
                [("integration-v2", "retired"), ("integration-v3", "published")],
            )
        repository = PostgresKnowledgeRepository(self.connection)
        self.assertTrue(repository.readiness(corpus("integration-v3")).ready)
        result = repository.exact_cosine_search([0, 0, 1], "embed-v1", 1)
        self.assertEqual(len(result), 1)

        publisher.rollback("integration-v2")
        self.assertTrue(repository.readiness(corpus("integration-v2")).ready)
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM knowledge_chunks WHERE id = %s", (first,))
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_schema_rejects_cross_version_embedding(self):
        import psycopg

        first = self._stage("fk-v1", "[1,0,0]")
        self._stage("fk-v2", "[0,1,0]")
        with self.connection.cursor() as cursor:
            cursor.execute("SAVEPOINT cross_version")
            with self.assertRaises(psycopg.errors.ForeignKeyViolation):
                cursor.execute(
                    """
                    INSERT INTO knowledge_embeddings
                        (chunk_id, embedding_model, embedding_dimension, embedding, corpus_version)
                    VALUES (%s, 'other-model', 3, '[1,0,0]'::vector, 'fk-v2')
                    """,
                    (first,),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT cross_version")
        self.connection.commit()

    def test_schema_migrations_are_idempotent(self):
        apply_schema(self.connection)
        apply_schema(self.connection)

    def test_staging_and_embedding_repository_interfaces_persist_real_rows(self):
        chunk_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())
        chunk = SimpleNamespace(
            chunk_id=chunk_id, document_id=document_id, parent_section_id=None,
            title="Title", heading_path=["Title"], content="Body", topics=["topic"],
            task_types=["reply"], relationship_stages=[], action_labels=["act"],
            applicable_when=[], not_applicable_when=[], knowledge_type="source_note",
            evidence_level="L4", review_status="approved", priority=1,
            source_path="knowledge/integration.md", source_sha256="sha",
            corpus_version="repository-v1", token_count=2,
        )
        build = SimpleNamespace(
            report=SimpleNamespace(
                corpus_version="repository-v1", embedding_model="embed-v1",
                embedding_dimension=3, embedding_normalization="l2",
                tokenizer_version="tokens-v1", chunker_version="chunks-v1",
            ),
            chunks=[SimpleNamespace(
                chunk=chunk,
                relationships=SimpleNamespace(
                    previous_chunk_id=None, next_chunk_id=None,
                ),
            )],
        )
        repository = PostgresKnowledgeRepository(self.connection)
        repository.write_staging_build(build, {chunk_id: {
            "A": "title topic act", "B": "title", "C": "body",
            "D": "knowledge integration md",
        }})
        repository.write_embeddings("repository-v1", "embed-v1", {chunk_id: [1, 0, 0]})
        CorpusPublisher(self.connection).publish("repository-v1")
        result = repository.exact_cosine_search([1, 0, 0], "embed-v1", 1)
        self.assertEqual(result[0]["chunk_id"], chunk_id)
        self.assertEqual(result[0]["metadata"]["token_count"], 2)
        filtered = repository.exact_cosine_search(
            [1, 0, 0], "embed-v1", 1, {"task_types": ["reply"]}
        )
        self.assertEqual(filtered[0]["chunk_id"], chunk_id)
        excluded = repository.exact_cosine_search(
            [1, 0, 0], "embed-v1", 1, {"task_types": ["invitation"]}
        )
        self.assertEqual(excluded, [])

    def test_retrieval_trace_round_trip_and_failure_update(self):
        repository = PostgresKnowledgeRepository(self.connection)
        request_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        repository.write_trace(
            {
                "request_id": request_id,
                "conversation_id": "integration-trace",
                "skill_version": "1.0.0",
                "skill_sha256": "a" * 64,
                "corpus_version": "2026.07.1",
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "reranker_model": "BAAI/bge-reranker-base",
                "scene_snapshot": {
                    "task_type_id": "reply",
                    "recommended_action_id": "respond",
                    "current_event_id": "b" * 64,
                    "user_goal_id": "c" * 64,
                    "active_skill_scenario_ids": ["boundary"],
                    "current_event_sample": "受控调试样本",
                },
                "retrieval_plan": {
                    "query_ids": ["d" * 64],
                    "required_topic_ids": ["boundary"],
                    "excluded_topic_ids": ["manipulation"],
                    "hard_filter_ids": {"review_status": ["approved"]},
                    "soft_preference_ids": {},
                    "query_samples": ["低压力回复"],
                },
                "evidence_assessment": {
                    "status": "sufficient",
                    "selected_chunk_ids": [chunk_id],
                },
                "fallback_reason": None,
                "latency_ms": {"rerank_latency_ms": 1.5},
                "lexical_candidates": [],
                "vector_candidates": [],
                "fused_candidates": [],
                "reranked_candidates": [],
                "selected_chunks": [{"chunk_id": chunk_id, "score": 0.9}],
            }
        )
        repository.mark_trace_failure(request_id, "generation_failed", 2.5)
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT embedding_model, reranker_model, scene_snapshot,
                       retrieval_plan, fallback_reason,
                       latency_ms->>'generation_latency_ms'
                FROM retrieval_traces WHERE request_id = %s
                """,
                (request_id,),
            )
            trace = cursor.fetchone()
            cursor.execute(
                """
                SELECT stage, chunk_id::text, rank, score
                FROM retrieval_trace_candidates WHERE request_id = %s
                """,
                (request_id,),
            )
            candidate = cursor.fetchone()
        self.assertEqual(trace[0], "BAAI/bge-small-zh-v1.5")
        self.assertEqual(trace[1], "BAAI/bge-reranker-base")
        self.assertEqual(trace[2]["task_type_id"], "reply")
        self.assertEqual(trace[2]["recommended_action_id"], "respond")
        self.assertEqual(trace[2]["current_event_sample"], "受控调试样本")
        self.assertEqual(trace[3]["required_topic_ids"], ["boundary"])
        self.assertEqual(trace[3]["query_samples"], ["低压力回复"])
        self.assertEqual(trace[4], "generation_failed")
        self.assertEqual(float(trace[5]), 2.5)
        self.assertEqual(candidate, ("selected_chunks", chunk_id, 1, 0.9))


if __name__ == "__main__":
    unittest.main()
