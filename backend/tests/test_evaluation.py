import json
import tempfile
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.gate_support import (
    atomic_json,
    build_runtime_state,
    prepare_outputs,
    publish_outputs,
)
from evaluation.metrics import evaluate
from evaluation.run_live_gate import (
    _latency_summary,
    _source_coverage,
    _trace_completeness,
)
from evaluation.run_scene_plan_gate import evaluate_plans, run_cases
from evaluation.build_golden_dataset import build_cases
from app.knowledge.ingestion import build_corpus
from app.knowledge.governance.corpus_version import CURRENT_CORPUS_VERSION
from app.knowledge.ingestion.metadata_enricher import CONTROLLED_TOPICS


DATASET = Path(__file__).parents[1] / "evaluation" / "golden_dataset_v1.jsonl"


class OfflineEvaluationTests(unittest.TestCase):
    def test_scene_plan_metrics_count_missing_predictions_as_wrong(self):
        cases = [
            {
                "id": "one",
                "category": "reply",
                "required_topics": ["boundary", "reciprocity"],
                "expected_task_type": "reply",
                "expected_action_direction": "respond",
            },
            {
                "id": "two",
                "category": "reply",
                "required_topics": [],
                "expected_task_type": "clarify",
                "expected_action_direction": "clarify",
            },
        ]
        results = evaluate_plans(
            cases,
            {
                "one": {
                    "required_topics": ["boundary"],
                    "task_type": "reply",
                    "action_direction": "respond",
                }
            },
        )

        self.assertEqual(results["case_count"], 2)
        self.assertEqual(results["required_topic_coverage"], 0.25)
        self.assertEqual(results["task_type_accuracy"], 0.5)
        self.assertEqual(results["action_direction_accuracy"], 0.5)
        self.assertEqual(
            results["required_topic_coverage_by_category"],
            {"reply": 0.25},
        )
        self.assertEqual(
            evaluate_plans(
                [cases[1]],
                {
                    "two": {
                        "required_topics": [],
                        "task_type": "clarify",
                        "action_direction": "clarify",
                    }
                },
            )["required_topic_coverage"],
            1.0,
        )

    def test_scene_plan_runner_keeps_partial_when_a_later_case_fails(self):
        cases = [
            {"id": "one", "query": "first"},
            {"id": "two", "query": "second"},
        ]
        calls = 0

        def analyze(state):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("model unavailable")
            return {
                "scene_and_retrieval_plan": {
                    "scene": {
                        "task_type": "reply",
                        "recommended_action": "respond",
                        "active_skill_scenarios": ["explicit_rejection"],
                    },
                    "retrieval": {
                        "queries": ["respect the rejection"],
                        "required_topics": ["boundary"],
                        "optional_topics": [],
                        "excluded_topics": ["aggressive_pursuit"],
                    },
                }
            }

        runtime = SimpleNamespace(
            graph_lock=nullcontext(),
            nodes=SimpleNamespace(build_context=lambda state: {}, analyze_scene_and_plan=analyze),
        )
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "predictions.partial.json"
            with self.assertRaises(RuntimeError):
                run_cases(cases, runtime, partial)
            saved = json.loads(partial.read_text(encoding="utf-8"))
            self.assertEqual(list(saved), ["one"])
            self.assertEqual(saved["one"]["queries"], ["respect the rejection"])
            self.assertEqual(
                saved["one"]["excluded_topics"], ["aggressive_pursuit"]
            )
            self.assertEqual(
                saved["one"]["active_skill_scenarios"],
                ["explicit_rejection"],
            )

    def test_prepare_outputs_removes_stale_gate_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("results", "predictions", "partial")]
            for path in paths:
                path.write_text("stale", encoding="utf-8")
            prepare_outputs(*paths)
            self.assertTrue(all(not path.exists() for path in paths))

    def test_atomic_json_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            original_replace = Path.replace
            attempts = 0

            def replace(path, target):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise PermissionError("temporarily locked")
                return original_replace(path, target)

            with patch.object(Path, "replace", autospec=True, side_effect=replace):
                with patch("evaluation.gate_support.time.sleep") as sleep:
                    atomic_json(output, {"complete": True})

            self.assertEqual(attempts, 2)
            sleep.assert_called_once_with(0.05)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"complete": True},
            )

    def test_atomic_json_reraises_after_three_permission_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            with patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=PermissionError("locked"),
            ) as replace:
                with patch("evaluation.gate_support.time.sleep") as sleep:
                    with self.assertRaises(PermissionError):
                        atomic_json(output, {})

            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_atomic_json_does_not_retry_other_os_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            with patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=OSError("disk full"),
            ) as replace:
                with patch("evaluation.gate_support.time.sleep") as sleep:
                    with self.assertRaises(OSError):
                        atomic_json(output, {})

            replace.assert_called_once()
            sleep.assert_not_called()

    def test_latency_summary_reports_nearest_rank_p50_and_p95(self):
        self.assertEqual(
            _latency_summary(
                [
                    {"lexical_retrieval_latency_ms": 1},
                    {"lexical_retrieval_latency_ms": 2},
                    {"lexical_retrieval_latency_ms": 3},
                    {"lexical_retrieval_latency_ms": 4},
                ]
            ),
            {"lexical_retrieval_latency_ms": {"p50": 2.0, "p95": 4.0}},
        )
        self.assertEqual(
            _latency_summary([{"lexical": None}, {"vector": 3}]),
            {"vector": {"p50": 3.0, "p95": 3.0}},
        )
        self.assertEqual(_latency_summary([]), {})

    def test_golden_dataset_has_120_complete_deidentified_cases_across_14_categories(self):
        cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
        labels = {
            "expected_task_type",
            "required_topics",
            "excluded_topics",
            "gold_chunks",
            "acceptable_chunks",
            "forbidden_chunks",
            "expected_action_direction",
        }
        self.assertEqual(len(cases), 120)
        self.assertEqual(len({case["id"] for case in cases}), 120)
        self.assertEqual(len(Counter(case["category"] for case in cases)), 14)
        self.assertTrue(all(labels <= case.keys() for case in cases))
        self.assertEqual(len({case["query"] for case in cases}), 120)

    def test_chunk_annotations_exist_in_corpus_and_are_pairwise_disjoint(self):
        cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
        corpus_ids = {
            item.chunk.chunk_id
            for item in build_corpus(CURRENT_CORPUS_VERSION).chunks
        }
        for case in cases:
            gold = set(case["gold_chunks"])
            acceptable = set(case["acceptable_chunks"])
            forbidden = set(case["forbidden_chunks"])
            self.assertFalse(gold & acceptable, case["id"])
            self.assertFalse(gold & forbidden, case["id"])
            self.assertFalse(acceptable & forbidden, case["id"])
            self.assertLessEqual(gold | acceptable | forbidden, corpus_ids, case["id"])

    def test_golden_topics_are_validated_against_the_corpus_ontology(self):
        self.assertEqual(len(build_cases()), 120)
        corpus_topics = {
            topic
            for item in build_corpus(CURRENT_CORPUS_VERSION).chunks
            for topic in item.chunk.topics
        }
        self.assertLessEqual(corpus_topics, CONTROLLED_TOPICS)
        chunks = {
            item.chunk.chunk_id: item.chunk
            for item in build_corpus(CURRENT_CORPUS_VERSION).chunks
        }
        for case in build_cases():
            if not case["gold_chunks"]:
                continue
            evidence_topics = set().union(
                *(
                    set(chunks[chunk_id].topics)
                    for chunk_id in case["gold_chunks"] + case["acceptable_chunks"]
                )
            )
            self.assertTrue(set(case["excluded_topics"]).isdisjoint(evidence_topics))

    def test_source_coverage_fails_closed_for_wrong_corpus_version(self):
        self.assertEqual(_source_coverage("wrong-version", set()), 0)

    def test_source_coverage_requires_ingested_sources_in_live_corpus(self):
        manifest = __import__("yaml").safe_load(
            (DATASET.parents[2] / "trellis/retrieval-system-refactor/source-manifest.yaml")
            .read_text(encoding="utf-8")
        )
        ingested = {
            source["path"] for source in manifest["sources"]
            if source["required"] and source["status"] == "ingested"
        }
        self.assertEqual(_source_coverage(CURRENT_CORPUS_VERSION, ingested), 1)
        self.assertLess(_source_coverage(CURRENT_CORPUS_VERSION, set()), 1)

    def test_trace_completeness_uses_all_expected_requests_as_denominator(self):
        self.assertEqual(_trace_completeness(["a", "b"], 2, 2), 1)
        self.assertEqual(_trace_completeness(["a", "b"], 1, 1), 0)
        self.assertEqual(_trace_completeness(["a", "b"], 2, 1), 0.5)
        self.assertEqual(_trace_completeness(["a", "a"], 1, 1), 0)

    def test_gate_results_are_published_last_as_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.json"
            predictions = root / "predictions.json"
            partial = root / "predictions.partial.json"
            partial.write_text("partial", encoding="utf-8")
            with patch(
                "evaluation.gate_support.atomic_json",
                side_effect=[None, OSError("results write failed")],
            ) as write:
                with self.assertRaises(OSError):
                    publish_outputs(results, predictions, partial, {}, {})
            self.assertEqual(
                [call.args[0] for call in write.call_args_list],
                [predictions, results],
            )
            self.assertFalse(results.exists())
            self.assertTrue(partial.exists())

    def test_live_gate_runtime_input_contains_no_golden_labels(self):
        state = build_runtime_state("用户输入", "case-1")
        serialized = json.dumps(state, ensure_ascii=False)
        self.assertNotIn("required_topics", serialized)
        self.assertNotIn("expected_", serialized)
        for label in ("gold_chunks", "acceptable_chunks", "forbidden_chunks", "category", "selector"):
            self.assertNotIn(label, serialized)
        self.assertEqual(state["user_message"], "用户输入")

    def test_hard_negative_pairs_require_opposite_actions_for_the_same_keyword(self):
        cases = [
            json.loads(line)
            for line in DATASET.read_text(encoding="utf-8").splitlines()
            if '"category": "Hard Negative"' in line
        ]
        self.assertEqual(len(cases), 8)
        for positive, negative in zip(cases[::2], cases[1::2]):
            positive_keyword = positive["query"].split("“", 1)[1].split("”", 1)[0]
            negative_keyword = negative["query"].split("“", 1)[1].split("”", 1)[0]
            self.assertEqual(positive_keyword, negative_keyword)
            self.assertNotEqual(
                positive["expected_action_direction"],
                negative["expected_action_direction"],
            )
            self.assertNotEqual(positive["gold_chunks"], negative["gold_chunks"])

    def test_metrics_are_reproducible_and_do_not_claim_gate_success(self):
        cases = [
            {
                "id": "one",
                "expected_task_type": "reply",
                "expected_action_direction": "respond",
                "required_topics": ["boundary"],
                "gold_chunks": ["gold"],
                "acceptable_chunks": ["ok"],
                "forbidden_chunks": ["bad"],
            },
            {
                "id": "two",
                "expected_task_type": "clarify",
                "expected_action_direction": "clarify",
                "required_topics": [],
                "gold_chunks": [],
                "acceptable_chunks": [],
                "forbidden_chunks": ["bad"],
            },
        ]
        predictions = {
            "one": {
                "ranked_chunks": ["ok", "gold"],
                "baseline_ranked_chunks": ["noise", "ok"],
                "topics": ["boundary"],
                "task_type": "reply",
                "action_direction": "respond",
                "no_evidence": False,
            },
            "two": {
                "ranked_chunks": [],
                "no_evidence": True,
                "task_type": "clarify",
                "action_direction": "clarify",
            },
        }
        result = evaluate(cases, predictions)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["recall_at_20"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["action_direction_accuracy"], 1.0)
        self.assertNotIn("passed", result)

    def test_action_accuracy_ignores_unlabelled_cases_and_missing_prediction_is_wrong(self):
        labelled = {
            "id": "labelled",
            "expected_task_type": "reply",
            "expected_action_direction": "respond",
            "required_topics": [],
            "gold_chunks": [],
            "acceptable_chunks": [],
            "forbidden_chunks": [],
        }
        unlabelled = {**labelled, "id": "unlabelled"}
        unlabelled.pop("expected_action_direction")

        result = evaluate(
            [labelled, unlabelled],
            {"labelled": {"action_direction": "respond"}},
        )

        self.assertEqual(result["action_direction_accuracy"], 1.0)
        self.assertEqual(evaluate([labelled], {})["action_direction_accuracy"], 0.0)

    def test_reranker_metric_is_unavailable_without_baseline_pairs(self):
        case = {
            "id": "one",
            "expected_task_type": "reply",
            "required_topics": [],
            "gold_chunks": ["gold"],
            "acceptable_chunks": [],
            "forbidden_chunks": [],
        }
        self.assertIsNone(evaluate([case], {"one": {"ranked_chunks": ["gold"]}})["reranker_win_rate"])

    def test_recall_uses_top_20_retrieval_while_ndcg_uses_reranked_output(self):
        case = {
            "id": "one",
            "expected_task_type": "reply",
            "required_topics": [],
            "gold_chunks": ["gold"],
            "acceptable_chunks": [],
            "forbidden_chunks": [],
        }
        result = evaluate(
            [case],
            {
                "one": {
                    "retrieved_chunks": ["noise"] * 14 + ["gold"],
                    "ranked_chunks": ["noise"],
                }
            },
        )
        self.assertEqual(result["recall_at_20"], 1.0)
        self.assertEqual(result["ndcg_at_10"], 0.0)

    def test_empty_predictions_cannot_pass_no_evidence_gate(self):
        cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
        result = evaluate(cases, {})
        self.assertLess(result["no_evidence_accuracy"], 0.90)
        self.assertEqual(result["evidence_detection_recall"], 0.0)
        self.assertEqual(result["no_evidence_recall"], 0.0)

    def test_forbidden_rejection_uses_only_explicitly_annotated_cases(self):
        cases = [
            {
                "id": "annotated",
                "expected_task_type": "reply",
                "required_topics": [],
                "gold_chunks": ["gold"],
                "acceptable_chunks": [],
                "forbidden_chunks": ["bad"],
            },
            {
                "id": "not-annotated",
                "expected_task_type": "reply",
                "required_topics": [],
                "gold_chunks": ["gold"],
                "acceptable_chunks": [],
                "forbidden_chunks": [],
            },
        ]
        predictions = {
            "annotated": {"ranked_chunks": ["bad"]},
            "not-annotated": {"ranked_chunks": ["gold"]},
        }
        result = evaluate(cases, predictions)
        self.assertEqual(result["forbidden_annotated_case_count"], 1)
        self.assertEqual(result["forbidden_chunk_rejection_rate"], 0)

    def test_context_accuracy_uses_action_not_only_task_type(self):
        cases = [
            {
                "id": "context-a",
                "category": "Hard Negative",
                "expected_task_type": "上下文判别",
                "expected_action_direction": "继续互动",
                "required_topics": [],
                "gold_chunks": ["continue"],
                "acceptable_chunks": [],
                "forbidden_chunks": [],
            },
            {
                "id": "context-b",
                "category": "Hard Negative",
                "expected_task_type": "上下文判别",
                "expected_action_direction": "停止推进",
                "required_topics": [],
                "gold_chunks": ["stop"],
                "acceptable_chunks": [],
                "forbidden_chunks": [],
            },
        ]
        predictions = {
            case["id"]: {
                "task_type": "上下文判别",
                "action_direction": "继续互动",
                "no_evidence": False,
            }
            for case in cases
        }
        self.assertEqual(evaluate(cases, predictions)["context_dependent_query_accuracy"], 0.0)
        perfect = {
            case["id"]: {
                "task_type": "上下文判别",
                "action_direction": case["expected_action_direction"],
                "ranked_chunks": case["gold_chunks"],
                "no_evidence": False,
            }
            for case in cases
        }
        self.assertEqual(evaluate(cases, perfect)["context_dependent_query_accuracy"], 1.0)

    def test_no_evidence_requires_explicit_flag_and_empty_retrieval(self):
        case = {
            "id": "no-evidence",
            "category": "信息不足",
            "expected_task_type": "信息澄清",
            "expected_action_direction": "先澄清",
            "required_topics": [],
            "gold_chunks": [],
            "acceptable_chunks": [],
            "forbidden_chunks": [],
        }
        self.assertEqual(evaluate([case], {})["no_evidence_accuracy"], 0.0)
        contradictory = {"no-evidence": {"no_evidence": True, "ranked_chunks": ["noise"]}}
        self.assertEqual(evaluate([case], contradictory)["no_evidence_accuracy"], 0.0)
        explicit = {"no-evidence": {"no_evidence": True, "ranked_chunks": []}}
        self.assertEqual(evaluate([case], explicit)["no_evidence_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
