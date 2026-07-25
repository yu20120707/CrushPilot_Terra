import json
import unittest
from collections import Counter
from pathlib import Path

from evaluation.metrics import evaluate
from evaluation.run_live_gate import _source_coverage, build_runtime_state
from evaluation.build_golden_dataset import build_cases
from app.knowledge.ingestion import build_corpus
from app.knowledge.ingestion.metadata_enricher import CONTROLLED_TOPICS


DATASET = Path(__file__).parents[1] / "evaluation" / "golden_dataset_v1.jsonl"


class OfflineEvaluationTests(unittest.TestCase):
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
        corpus_ids = {item.chunk.chunk_id for item in build_corpus("2026.07.5").chunks}
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
            for item in build_corpus("2026.07.5").chunks
            for topic in item.chunk.topics
        }
        self.assertLessEqual(corpus_topics, CONTROLLED_TOPICS)
        chunks = {
            item.chunk.chunk_id: item.chunk
            for item in build_corpus("2026.07.5").chunks
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
        self.assertEqual(_source_coverage("wrong-version"), 0)

    def test_live_gate_runtime_input_contains_no_golden_labels(self):
        state = build_runtime_state("用户输入", "case-1")
        serialized = json.dumps(state, ensure_ascii=False)
        self.assertNotIn("required_topics", serialized)
        self.assertNotIn("expected_", serialized)
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
                "required_topics": ["boundary"],
                "gold_chunks": ["gold"],
                "acceptable_chunks": ["ok"],
                "forbidden_chunks": ["bad"],
            },
            {
                "id": "two",
                "expected_task_type": "clarify",
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
                "no_evidence": False,
            },
            "two": {"ranked_chunks": [], "no_evidence": True, "task_type": "clarify"},
        }
        result = evaluate(cases, predictions)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["recall_at_20"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertNotIn("passed", result)

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
