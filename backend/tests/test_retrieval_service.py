import unittest

from app.knowledge.domain.models import RetrievalPlan, SceneSnapshot
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.retrieval.lexical_retriever import LexicalRetriever, _or_tsquery
from app.knowledge.retrieval.metadata_enricher import (
    enrich_metadata,
    isolate_bad_candidates,
)
from app.knowledge.retrieval.reranker import RerankerUnavailable
from app.knowledge.retrieval.service import RetrievalService, _fit_budget
from app.knowledge.retrieval.vector_retriever import VectorRetriever, VectorUnavailable
from app.knowledge.retrieval.tokenizer import build_topic_tokens


def chunk(chunk_id, *, topic="boundary", content=None, document_id=None):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id or chunk_id,
        "title": chunk_id,
        "heading_path": [],
        "content": content or chunk_id,
        "topics": [topic],
        "action_labels": [],
        "knowledge_type": "principle",
        "token_count": 1,
    }


class FakeRetriever:
    def __init__(self, values, calls):
        self.values = values
        self.calls = calls

    def search(self, query, *args, hard_filters=None, **kwargs):
        self.calls.append((query, kwargs["limit"]))
        value = self.values[query.splitlines()[0]]
        if isinstance(value, Exception):
            raise value
        return value


class FilteringRetriever(FakeRetriever):
    def search(self, query, *args, hard_filters, **kwargs):
        self.calls.append((query, hard_filters))
        return self.values[query.splitlines()[0]]


class FakeReranker:
    model_name = "fake-reranker"

    def __init__(self, calls, unavailable=False):
        self.calls = calls
        self.unavailable = unavailable

    def score(self, scene, query, documents):
        self.calls.append(("rerank", len(documents)))
        if self.unavailable:
            raise RerankerUnavailable()
        return list(reversed(range(len(documents))))


class TraceWriter:
    def __init__(self, fail=False):
        self.traces = []
        self.fail = fail

    def write_trace(self, trace):
        if self.fail:
            raise OSError("trace down")
        self.traces.append(trace)


class ContextLoader:
    def __init__(self, values):
        self.values = values
        self.selected = None

    def load_expansion_context(self, selected):
        self.selected = selected
        return self.values


class FlexibleMetrics:
    def __init__(self):
        self.values = []

    def observe(self, name, value):
        self.values.append((name, value))


class FakeCursor:
    description = [
        type("Column", (), {"name": name})
        for name in (
            "chunk_id",
            "document_id",
            "parent_section_id",
            "title",
            "heading_path",
            "content",
            "metadata",
            "score",
        )
    ]

    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()

    def cursor(self):
        return self.fake_cursor


def plan():
    return RetrievalPlan(
        queries=["primary", "secondary"],
        required_topics=["boundary"],
        optional_topics=[],
        excluded_topics=["aggressive"],
        hard_filters={},
        soft_preferences={},
    )


def scene():
    return SceneSnapshot(
        task_type="boundary",
        recommended_action="stop",
        current_event="event",
        user_goal="goal",
        known_facts=[],
        uncertain_inferences=[],
        key_unknowns=[],
        counterpart_signals=[],
        explicit_boundaries=["stop"],
        active_skill_scenarios=[],
        confidence=1,
    )


class RetrievalServiceTests(unittest.TestCase):
    def run_service(self, lexical_values, vector_values, *, unavailable=False):
        lexical_calls = []
        vector_calls = []
        reranker_calls = []
        writer = TraceWriter()
        service = RetrievalService(
            FakeRetriever(lexical_values, lexical_calls),
            FakeRetriever(vector_values, vector_calls),
            FakeReranker(reranker_calls, unavailable),
            trace_writer=writer,
        )
        result = service.retrieve(
            request_id="req",
            conversation_id="conv",
            skill_version="1",
            skill_sha256="sha",
            corpus_version="corpus",
            scene=scene(),
            plan=plan(),
            embedding_model="embedding",
        )
        return result, lexical_calls, vector_calls, reranker_calls, writer

    def test_runs_all_stages_in_order_with_limits_and_trace(self):
        values = {
            "primary": [chunk(str(index)) for index in range(25)],
            "secondary": [chunk("secondary")],
        }
        result, lexical, vector, reranker, writer = self.run_service(values, values)
        self.assertEqual(lexical, [("primary", 20), ("secondary", 20)])
        self.assertEqual(
            vector,
            [
                ("primary\n边界 同意 越界 施压", 20),
                ("secondary\n边界 同意 越界 施压", 20),
            ],
        )
        self.assertEqual(reranker, [("rerank", 15)])
        self.assertLessEqual(len(result.chunks), 6)
        self.assertEqual(result.assessment.status, "sufficient")
        self.assertEqual(len(writer.traces), 1)
        self.assertEqual(
            set(result.trace.latency_ms),
            {
                "lexical_retrieval_latency_ms",
                "vector_retrieval_latency_ms",
                "fusion_latency_ms",
                "rerank_latency_ms",
                "diversity_latency_ms",
                "expansion_latency_ms",
                "evidence_gate_latency_ms",
            },
        )

    def test_vector_and_reranker_failures_degrade_and_bad_chunks_are_isolated(self):
        lexical = {
            "primary": [
                {"chunk_id": "bad", "content": "broken"},
                chunk("good"),
                chunk("excluded", topic="aggressive"),
            ],
            "secondary": [],
        }
        vector = {
            "primary": VectorUnavailable("down"),
            "secondary": [],
        }
        result, _, _, reranker, _ = self.run_service(
            lexical, vector, unavailable=True
        )
        self.assertEqual(reranker, [("rerank", 2)])
        self.assertEqual([item["chunk_id"] for item in result.chunks], ["good"])
        self.assertEqual(
            result.trace.fallback_reason,
            "vector_unavailable,reranker_unavailable",
        )

    def test_diversity_backfills_after_high_ranked_candidates_are_excluded(self):
        candidates = [
            chunk(f"excluded-{index}", topic="aggressive")
            for index in range(6)
        ] + [chunk("safe-7"), chunk("safe-8")]
        values = {"primary": candidates, "secondary": []}

        for unavailable in (False, True):
            with self.subTest(reranker_unavailable=unavailable):
                result, *_ = self.run_service(
                    values, values, unavailable=unavailable
                )
                self.assertEqual(
                    [item["chunk_id"] for item in result.chunks],
                    ["safe-7", "safe-8"],
                )
                self.assertEqual(len(result.trace.reranked_candidates), 8)
                self.assertEqual(
                    result.trace.fallback_reason,
                    "reranker_unavailable" if unavailable else None,
                )

    def test_new_kb_suppresses_conflicting_original_but_not_better_matched_original(self):
        newer = chunk("new", content="new stance")
        newer.update({"source_collection": "new_kb", "source_priority": 100, "decision_key": "boundary", "stance": "new"})
        older = chunk("old", content="old stance")
        older.update({"source_collection": "original", "source_priority": 50, "decision_key": "boundary", "stance": "old"})
        values = {"primary": [newer, older], "secondary": []}
        result, *_ = self.run_service(values, values)
        self.assertEqual([item["chunk_id"] for item in result.chunks], ["new"])
        suppressed = next(item for item in result.trace.reranked_candidates if item["chunk_id"] == "old")
        self.assertTrue(suppressed["suppressed_by_new_kb"])

    def test_same_topic_with_distinct_decision_keys_is_not_suppressed(self):
        newer = chunk("new", content="new guidance")
        newer.update({"source_collection": "new_kb", "source_priority": 100, "decision_key": "ending a relationship", "stance": "new"})
        older = chunk("old", content="old guidance")
        older.update({"source_collection": "original", "source_priority": 50, "decision_key": "recovering after a breakup", "stance": "old"})
        values = {"primary": [newer, older], "secondary": []}
        result, *_ = self.run_service(values, values)
        self.assertEqual({item["chunk_id"] for item in result.chunks}, {"new", "old"})
        old_trace = next(item for item in result.trace.reranked_candidates if item["chunk_id"] == "old")
        self.assertFalse(old_trace["suppressed_by_new_kb"])

    def test_insufficient_evidence_is_not_injected_into_generation(self):
        unrelated = chunk("unrelated", topic="invitation")
        values = {"primary": [unrelated], "secondary": []}
        result, *_ = self.run_service(values, values)
        self.assertEqual(result.assessment.status, "insufficient")
        self.assertEqual(result.assessment.selected_chunk_ids, [])
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.trace.selected_chunks, [])

    def test_expansion_is_after_selection_and_respects_budget(self):
        parent = chunk("parent", content="p")
        core = chunk("core", content="c")
        core["parent_section_id"] = "parent"
        parent["token_count"] = 3499
        core["token_count"] = 2
        values = {"primary": [core, parent], "secondary": []}
        result, *_ = self.run_service(values, {"primary": [], "secondary": []})
        self.assertEqual(result.chunks[0]["chunk_id"], "core")
        self.assertEqual(result.chunks[0]["expanded_context"], [])

    def test_context_loader_expands_real_parent_and_neighbor_ids(self):
        core = chunk("550e8400-e29b-41d4-a716-446655440000")
        core.update(
            {
                "parent_section_id": "550e8400-e29b-41d4-a716-446655440001",
                "previous_chunk_id": "550e8400-e29b-41d4-a716-446655440002",
            }
        )
        contexts = {
            "550e8400-e29b-41d4-a716-446655440001": {
                **chunk("550e8400-e29b-41d4-a716-446655440001"),
                "context_kind": "parent",
            },
            "550e8400-e29b-41d4-a716-446655440002": {
                **chunk("550e8400-e29b-41d4-a716-446655440002"),
                "context_kind": "neighbor",
            },
        }
        loader = ContextLoader(contexts)
        values = {"primary": [core], "secondary": []}
        service = RetrievalService(
            FakeRetriever(values, []),
            FakeRetriever({"primary": [], "secondary": []}, []),
            FakeReranker([]),
            context_loader=loader,
        )
        result = service.retrieve(
            request_id="req",
            conversation_id="conv",
            skill_version="1",
            skill_sha256="sha",
            corpus_version="corpus",
            scene=scene(),
            plan=plan(),
        )
        self.assertEqual(loader.selected[0]["chunk_id"], core["chunk_id"])
        self.assertEqual(
            [item["context_kind"] for item in result.chunks[0]["expanded_context"]],
            ["parent", "neighbor"],
        )

    def test_not_applicable_scenario_is_filtered_before_selection(self):
        blocked = chunk("blocked")
        blocked["not_applicable_when"] = ["cold_contact"]
        values = {"primary": [blocked], "secondary": []}
        blocked_scene = scene().model_copy(
            update={"active_skill_scenarios": ["cold_contact"]}
        )
        service = RetrievalService(
            FakeRetriever(values, []),
            FakeRetriever({"primary": [], "secondary": []}, []),
            FakeReranker([]),
        )
        result = service.retrieve(
            request_id="req",
            conversation_id="conv",
            skill_version="1",
            skill_sha256="sha",
            corpus_version="corpus",
            scene=blocked_scene,
            plan=plan(),
        )
        self.assertEqual(result.chunks, [])
        self.assertEqual(result.assessment.status, "insufficient")

    def test_all_reranker_exceptions_degrade_with_warning(self):
        class BrokenReranker(FakeReranker):
            def score(self, scene, query, documents):
                raise ValueError("bad custom response")

        values = {"primary": [chunk("good")], "secondary": []}
        service = RetrievalService(
            FakeRetriever(values, []),
            FakeRetriever({"primary": [], "secondary": []}, []),
            BrokenReranker([]),
        )
        with self.assertLogs(
            "app.knowledge.retrieval.service", level="WARNING"
        ):
            result = service.retrieve(
                request_id="req",
                conversation_id="conv",
                skill_version="1",
                skill_sha256="sha",
                corpus_version="corpus",
                scene=scene(),
                plan=plan(),
            )
        self.assertEqual(result.trace.fallback_reason, "reranker_unavailable")

    def test_budget_keeps_low_score_core_before_example_or_context(self):
        core = chunk("core")
        core.update(
            {
                "token_count": 4,
                "rerank_score": 0.1,
                "expanded_context": [
                    {
                        **chunk("parent"),
                        "token_count": 4,
                        "context_kind": "parent",
                    }
                ],
            }
        )
        example = chunk("example")
        example.update(
            {
                "knowledge_type": "example",
                "token_count": 4,
                "rerank_score": 100,
                "expanded_context": [],
            }
        )
        selected = _fit_budget([example, core], 4, [])
        self.assertEqual([item["chunk_id"] for item in selected], ["core"])
        self.assertEqual(selected[0]["expanded_context"], [])

    def test_vector_failure_uses_lexical_and_records_fallback(self):
        lexical = {"primary": [chunk("good")], "secondary": []}
        vector = {
            "primary": VectorUnavailable("down"),
            "secondary": [],
        }
        result, *_ = self.run_service(lexical, vector)
        self.assertEqual([item["chunk_id"] for item in result.chunks], ["good"])
        self.assertEqual(result.trace.fallback_reason, "vector_unavailable")
        self.assertEqual(result.assessment.fallback_reason, "vector_unavailable")

    def test_no_candidates_skips_reranker_and_is_insufficient(self):
        values = {"primary": [], "secondary": []}
        result, _, _, reranker, _ = self.run_service(values, values)
        self.assertEqual(reranker, [])
        self.assertEqual(result.assessment.status, "insufficient")

    def test_trace_failure_does_not_fail_retrieval(self):
        values = {"primary": [chunk("good")], "secondary": []}
        metrics = FlexibleMetrics()
        service = RetrievalService(
            FakeRetriever(values, []),
            FakeRetriever({"primary": [], "secondary": []}, []),
            FakeReranker([]),
            trace_writer=TraceWriter(fail=True),
            metrics=metrics,
        )
        with self.assertLogs(
            "app.knowledge.retrieval.service", level="WARNING"
        ) as logs:
            result = service.retrieve(
                request_id="req",
                conversation_id="conv",
                skill_version="1",
                skill_sha256="sha",
                corpus_version="corpus",
                scene=scene(),
                plan=plan(),
            )
        self.assertEqual(result.chunks[0]["chunk_id"], "good")
        self.assertIn("trace write failed", logs.output[0])
        self.assertIn(("trace_write_failure_count", 1), metrics.values)

    def test_metrics_filters_threshold_and_trace_safety(self):
        values = {"primary": [chunk("good")], "secondary": []}
        lexical = FilteringRetriever(values, [])
        vector = FilteringRetriever(
            {"primary": [], "secondary": []}, []
        )
        metrics = RetrievalMetrics()
        service = RetrievalService(
            lexical,
            vector,
            FakeReranker([]),
            metrics=metrics,
            reranker_threshold=10,
        )
        configured_plan = plan().model_copy(
            update={
                "hard_filters": {"task_types": ["boundary"]},
                "soft_preferences": {"knowledge_type": ["principle"]},
            }
        )
        result = service.retrieve(
            request_id="req",
            conversation_id="conv",
            skill_version="1",
            skill_sha256="sha",
            corpus_version="corpus",
            scene=scene(),
            plan=configured_plan,
        )
        self.assertEqual(
            lexical.calls[0][1], {"task_types": ["boundary"]}
        )
        self.assertEqual(
            vector.calls[0][1], {"task_types": ["boundary"]}
        )
        self.assertEqual(result.assessment.status, "partially_sufficient")
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["selected_chunk_count"]["latest"], 1)
        self.assertEqual(snapshot["retrieval_partial_evidence_rate"]["latest"], 1)
        self.assertNotIn("content", result.trace.lexical_candidates[0])
        self.assertIn("content_id", result.trace.lexical_candidates[0])
        self.assertIn("score", result.trace.fused_candidates[0])
        self.assertNotIn("event", result.trace.scene_snapshot.values())
        self.assertEqual(
            result.trace.scene_snapshot["recommended_action_id"], "stop"
        )
        self.assertNotIn("primary", str(result.trace.retrieval_plan.values()))

    def test_trace_debug_explicitly_includes_controlled_samples(self):
        debug_scene = scene().model_copy(
            update={"current_event": "联系我 13812345678 或 a@example.com " + "很" * 300}
        )
        values = {"primary": [chunk("safe")], "secondary": []}
        service = RetrievalService(
            FakeRetriever(values, []),
            FakeRetriever(values, []),
            FakeReranker([]),
            trace_debug=True,
        )

        result = service.retrieve(
            request_id="trace-debug",
            conversation_id="conversation",
            skill_version="skill",
            skill_sha256="sha",
            corpus_version="corpus",
            scene=debug_scene,
            plan=plan(),
        )

        sample = result.trace.scene_snapshot["current_event_sample"]
        self.assertIn("[phone]", sample)
        self.assertIn("[email]", sample)
        self.assertNotIn("13812345678", sample)
        self.assertLessEqual(len(sample), 200)
        self.assertEqual(
            result.trace.retrieval_plan["query_samples"],
            ["primary", "secondary"],
        )


class LexicalRetrieverTests(unittest.TestCase):
    def test_required_topics_expand_to_shared_chinese_search_terms(self):
        tokens = build_topic_tokens(["reduce_pressure"])
        self.assertIn("降压", tokens)
        self.assertIn("留", tokens)
        self.assertIn("reduce", tokens)

    def test_query_tokens_are_or_joined_and_quote_escaped(self):
        self.assertEqual(
            _or_tsquery(["边界", "don't", "边界"]),
            "'边界' | 'don''t'",
        )

    def test_search_binds_required_query_excluded_and_limit_in_sql_order(self):
        connection = FakeConnection()

        LexicalRetriever(connection).search(
            "请不要联系",
            ["边界"],
            ["施压"],
            limit=50,
        )

        _, params = connection.fake_cursor.executed[0]
        sql, params = connection.fake_cursor.executed[0]
        self.assertIn("1.5 * ts_rank_cd(c.search_vector, tq.query, 2)", sql)
        self.assertEqual(params[0], ["边界"])
        self.assertIn("联系", params[1])
        self.assertIn("边界", params[2])
        self.assertEqual(params[3:], (["边界"], ["施压"], 20))

    def test_empty_query_does_not_touch_database(self):
        connection = FakeConnection()

        result = LexicalRetriever(connection).search("", [], [], limit=20)

        self.assertEqual(result, [])
        self.assertEqual(connection.fake_cursor.executed, [])

    def test_whitelisted_filters_are_parameterized_in_stable_order(self):
        connection = FakeConnection()
        LexicalRetriever(connection).search(
            "边界",
            [],
            [],
            hard_filters={
                "task_types": ["boundary"],
                "review_status": ["approved"],
                "relationship_stages": ["dating"],
            },
        )
        sql, params = connection.fake_cursor.executed[0]
        self.assertIn("c.review_status = ANY(%s)", sql)
        self.assertIn("metadata->'task_types'", sql)
        self.assertIn("metadata->'relationship_stages'", sql)
        self.assertEqual(
            params[5:8],
            (["dating"], ["approved"], ["boundary"]),
        )

    def test_unknown_or_empty_hard_filter_fails_closed(self):
        connection = FakeConnection()
        retriever = LexicalRetriever(connection)
        with self.assertRaises(ValueError):
            retriever.search("边界", [], [], hard_filters={"raw_sql": ["1=1"]})
        self.assertEqual(
            retriever.search(
                "边界", [], [], hard_filters={"task_types": []}
            ),
            [],
        )
        self.assertEqual(connection.fake_cursor.executed, [])


class VectorRetrieverTests(unittest.TestCase):
    def test_hard_filters_are_forwarded_to_repository(self):
        class Repository:
            def __init__(self):
                self.call = None

            def exact_cosine_search(self, embedding, model, **kwargs):
                self.call = (embedding, model, kwargs)
                return []

        repository = Repository()
        retriever = VectorRetriever(repository, lambda _: [1.0], "embed")
        hard_filters = {"task_types": ["boundary"]}
        retriever.search("query", hard_filters=hard_filters)
        self.assertEqual(
            repository.call,
            ([1.0], "embed", {"limit": 20, "hard_filters": hard_filters}),
        )


class MetadataEnricherTests(unittest.TestCase):
    def test_controlled_ontology_is_schema_complete_and_stop_wins_conflict(self):
        metadata = enrich_metadata(
            source_path="boundary.md",
            title="拒绝后停止推进",
            heading_path=["关系边界"],
            content="对方拒绝后不要继续邀约，先提供情绪支持。",
        )
        self.assertEqual(
            set(metadata),
            {
                "topics",
                "knowledge_type",
                "action_labels",
                "applicable_when",
                "not_applicable_when",
            },
        )
        self.assertIn("rejection", metadata["topics"])
        self.assertIn("relationship_progress", metadata["topics"])
        self.assertIn("emotional_support", metadata["topics"])
        self.assertEqual(metadata["action_labels"], ["stop"])

    def test_bad_candidate_is_isolated_with_warning(self):
        with self.assertLogs(
            "app.knowledge.retrieval.metadata_enricher", level="WARNING"
        ):
            result = isolate_bad_candidates(
                [{"chunk_id": "bad", "content": "missing document"}]
            )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
