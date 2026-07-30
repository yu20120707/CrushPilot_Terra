"""Evaluate the production scene planner without running retrieval or reranking."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from collections import defaultdict
from pathlib import Path

from app.knowledge.governance.corpus_version import CURRENT_CORPUS_VERSION
from app.bootstrap.runtime import AppRuntime
from app.core.config import Settings
from evaluation.gate_support import (
    DATASET,
    TRELIS,
    atomic_json,
    build_runtime_state,
    prepare_outputs,
    publish_outputs,
)


DEFAULT_OUTPUT = TRELIS / "scene-plan-gate-results.json"
DEFAULT_PREDICTIONS = TRELIS / "scene-plan-gate-predictions.json"


def evaluate_plans(cases: list[dict], predictions: dict[str, dict]) -> dict:
    topic_scores = []
    task_scores = []
    action_scores = []
    category_scores: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        if case["id"] not in predictions:
            topic_scores.append(0.0)
            task_scores.append(0.0)
            action_scores.append(0.0)
            category_scores[case["category"]].append(0.0)
            continue
        prediction = predictions.get(case["id"], {})
        expected_topics = set(case["required_topics"])
        predicted_topics = set(prediction.get("required_topics", []))
        topic_score = (
            len(expected_topics & predicted_topics) / len(expected_topics)
            if expected_topics
            else 1.0
        )
        topic_scores.append(topic_score)
        category_scores[case["category"]].append(topic_score)
        task_scores.append(
            float(prediction.get("task_type") == case["expected_task_type"])
        )
        action_scores.append(
            float(
                prediction.get("action_direction")
                == case["expected_action_direction"]
            )
        )
    count = len(cases)
    if not count:
        raise ValueError("at least one Golden Dataset case is required")
    return {
        "case_count": count,
        "required_topic_coverage": sum(topic_scores) / count,
        "task_type_accuracy": sum(task_scores) / count,
        "action_direction_accuracy": sum(action_scores) / count,
        "required_topic_coverage_by_category": {
            category: sum(scores) / len(scores)
            for category, scores in sorted(category_scores.items())
        },
    }


def run_cases(cases: list[dict], runtime: object, partial_output: Path) -> dict:
    predictions = {}
    for index, case in enumerate(cases, 1):
        print(f"case_start={index}/{len(cases)} id={case['id']}", flush=True)
        state = build_runtime_state(case["query"], f"scene-plan-{case['id']}")
        with runtime.graph_lock:
            state.update(runtime.nodes.build_context(state))
            state.update(runtime.nodes.analyze_scene_and_plan(state))
        plan = state["scene_and_retrieval_plan"]
        retrieval = plan["retrieval"]
        predictions[case["id"]] = {
            "queries": retrieval.get("queries", []),
            "required_topics": retrieval["required_topics"],
            "optional_topics": retrieval.get("optional_topics", []),
            "excluded_topics": retrieval.get("excluded_topics", []),
            "task_type": plan["scene"]["task_type"],
            "action_direction": plan["scene"]["recommended_action"],
            "active_skill_scenarios": plan["scene"].get(
                "active_skill_scenarios", []
            ),
        }
        atomic_json(partial_output, predictions)
        print(f"case_done={index}/{len(cases)} id={case['id']}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--corpus-version", default=CURRENT_CORPUS_VERSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--predictions-output", type=Path, default=DEFAULT_PREDICTIONS
    )
    args = parser.parse_args()
    partial_output = args.predictions_output.with_suffix(".partial.json")
    prepare_outputs(args.output, args.predictions_output, partial_output)

    os.environ["DEMO_MODE"] = "false"
    os.environ["DATABASE_URL"] = args.database_url
    os.environ["CORPUS_VERSION"] = args.corpus_version
    runtime = AppRuntime(Settings.from_env()).start()
    atexit.register(runtime.close)

    if runtime.readiness_errors:
        raise RuntimeError(f"production runtime is not ready: {runtime.readiness_errors}")

    cases = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
    ]
    predictions = run_cases(cases, runtime, partial_output)
    results = evaluate_plans(cases, predictions)
    publish_outputs(
        args.output,
        args.predictions_output,
        partial_output,
        results,
        predictions,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
