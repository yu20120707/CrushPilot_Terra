"""Small, dependency-free metrics for the retrieval Golden Dataset."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    return next((rank for rank, chunk in enumerate(ranked, 1) if chunk in relevant), None)


def _ndcg(ranked: list[str], gold: set[str], acceptable: set[str], limit: int = 10) -> float:
    gains = [2 if chunk in gold else 1 if chunk in acceptable else 0 for chunk in ranked[:limit]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(([2] * len(gold) + [1] * len(acceptable)), reverse=True)[:limit]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 1.0


def evaluate(cases: Iterable[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Evaluate predictions keyed by case id; missing predictions count as empty."""
    rows = []
    reranker_pairs = []
    evidence_detection = {False: [], True: []}
    context_results = []
    for case in cases:
        prediction = predictions.get(str(case["id"]), {})
        ranked = list(prediction.get("ranked_chunks", ()))
        retrieved = list(prediction.get("retrieved_chunks", ranked))
        gold, acceptable = set(case["gold_chunks"]), set(case["acceptable_chunks"])
        relevant = gold | acceptable
        first = _first_rank(ranked, relevant)
        required = set(case["required_topics"])
        returned_topics = set(prediction.get("topics", ()))
        forbidden = set(case["forbidden_chunks"])
        expected_no_evidence = not gold and not acceptable
        predicted_no_evidence = prediction.get("no_evidence")
        detection_correct = (
            isinstance(predicted_no_evidence, bool)
            and predicted_no_evidence == expected_no_evidence
        )
        if expected_no_evidence:
            detection_correct = detection_correct and not ranked
        evidence_detection[expected_no_evidence].append(float(detection_correct))
        if case.get("category") == "Hard Negative":
            context_results.append(
                bool(
                    prediction.get("action_direction")
                    == case["expected_action_direction"]
                    and gold.intersection(ranked[:20])
                )
            )
        rows.append(
            {
                "recall": (
                    len(gold.intersection(retrieved[:20])) / len(gold)
                    if gold else None
                ),
                "rr": (1 / first if first else 0.0) if relevant else None,
                "ndcg": _ndcg(ranked, gold, acceptable) if relevant else None,
                "topics": len(required.intersection(returned_topics)) / len(required) if required else 1.0,
                "forbidden": (
                    1.0 if forbidden.isdisjoint(ranked) else 0.0
                ) if forbidden else None,
                "task_type": float(
                    prediction.get("task_type") == case["expected_task_type"]
                ),
            }
        )
        if relevant and "baseline_ranked_chunks" in prediction:
            baseline = _first_rank(list(prediction["baseline_ranked_chunks"]), relevant) or math.inf
            reranked = _first_rank(ranked, relevant) or math.inf
            reranker_pairs.append(float(reranked < baseline))

    count = len(rows)
    if not count:
        raise ValueError("at least one Golden Dataset case is required")

    def mean(key: str) -> float:
        values = [row[key] for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else 0.0

    evidence_recall = (
        sum(evidence_detection[False]) / len(evidence_detection[False])
        if evidence_detection[False]
        else 0.0
    )
    no_evidence_recall = (
        sum(evidence_detection[True]) / len(evidence_detection[True])
        if evidence_detection[True]
        else 0.0
    )

    return {
        "case_count": count,
        "recall_at_20": mean("recall"),
        "mrr": mean("rr"),
        "ndcg_at_10": mean("ndcg"),
        "required_topic_coverage": mean("topics"),
        "task_type_accuracy": mean("task_type"),
        "forbidden_chunk_rejection_rate": mean("forbidden"),
        "forbidden_annotated_case_count": sum(
            row["forbidden"] is not None for row in rows
        ),
        # The release metric is scoped to explicitly annotated no-evidence cases.
        # Missing output is an error, not an implicit "no evidence" prediction.
        "no_evidence_accuracy": no_evidence_recall,
        "evidence_detection_recall": evidence_recall,
        "no_evidence_recall": no_evidence_recall,
        "reranker_win_rate": sum(reranker_pairs) / len(reranker_pairs) if reranker_pairs else None,
        "context_dependent_query_accuracy": (
            sum(
                all(context_results[index : index + 2])
                for index in range(0, len(context_results), 2)
            )
            / math.ceil(len(context_results) / 2)
            if context_results
            else None
        ),
    }
