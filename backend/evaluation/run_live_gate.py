"""Blind Golden evaluation through the production scene and retrieval nodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
import yaml

from evaluation.metrics import evaluate


DATASET = Path(__file__).with_name("golden_dataset_v1.jsonl")
ROOT = Path(__file__).parents[2]
TRELIS = ROOT / "trellis" / "retrieval-system-refactor"
DEFAULT_OUTPUT = TRELIS / "live-gate-results.json"
DEFAULT_PREDICTIONS = TRELIS / "live-gate-predictions.json"


def _source_coverage(expected_version: str) -> float:
    manifest = yaml.safe_load(
        (TRELIS / "source-manifest.yaml").read_text(encoding="utf-8")
    )
    if manifest.get("corpus_version") != expected_version:
        return 0
    required = [source for source in manifest["sources"] if source["required"]]
    covered = [
        source
        for source in required
        if source["status"] in {"ingested", "approved_excluded"}
        and (
            source["status"] == "approved_excluded"
            or source["chunk_count"] > 0
        )
    ]
    return len(covered) / len(required) if required else 0


def build_runtime_state(query: str, case_id: str) -> dict:
    return {
        "user_message": query,
        "device_id": "00000000-0000-0000-0000-000000000001",
        "conversation_id": f"golden-{case_id}",
        "relationship_id": None,
        "conversation_summary": None,
        "relationship_facts": [],
        "user_preferences": [],
        "messages": [{"role": "user", "content": query}],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--corpus-version", default="2026.07.5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--predictions-output", type=Path, default=DEFAULT_PREDICTIONS
    )
    args = parser.parse_args()

    # Import after setting runtime configuration so this is the production path.
    os.environ["DEMO_MODE"] = "false"
    os.environ["DATABASE_URL"] = args.database_url
    os.environ["CORPUS_VERSION"] = args.corpus_version
    from app import main as runtime

    if runtime._readiness_errors:
        raise RuntimeError(f"production runtime is not ready: {runtime._readiness_errors}")

    cases = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
    ]
    predictions: dict[str, dict] = {}
    trace_ids: list[str] = []
    boundary_hits: list[bool] = []

    with psycopg.connect(args.database_url) as read_connection:
        for index, case in enumerate(cases, 1):
            print(f"case_start={index}/{len(cases)} id={case['id']}", flush=True)
            state = build_runtime_state(case["query"], case["id"])
            with runtime.graph_lock:
                state.update(runtime.nodes.build_context(state))
                state.update(runtime.nodes.analyze_scene_and_plan(state))
                state.update(runtime.nodes.retrieve_evidence(state))
            trace_id = state["retrieval_trace_id"]
            trace_ids.append(trace_id)
            with read_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id::text
                    FROM retrieval_trace_candidates
                    WHERE request_id = %s AND stage = 'fused_candidates'
                    ORDER BY rank
                    """,
                    (trace_id,),
                )
                retrieved = [row[0] for row in cursor.fetchall() if row[0]]
            chunks = state.get("evidence_chunks", [])
            ranked = [chunk["chunk_id"] for chunk in chunks]
            plan = state["scene_and_retrieval_plan"]
            predictions[case["id"]] = {
                "retrieved_chunks": retrieved,
                "ranked_chunks": ranked,
                "baseline_ranked_chunks": retrieved,
                "topics": sorted(
                    {
                        topic
                        for chunk in chunks
                        for topic in chunk.get("topics", [])
                    }
                ),
                "task_type": plan["scene"]["task_type"],
                "action_direction": plan["scene"]["recommended_action"],
                "no_evidence": (
                    state["evidence_assessment"]["status"] == "insufficient"
                    and not chunks
                ),
            }
            partial_output = args.predictions_output.with_suffix(".partial.json")
            partial_temp = partial_output.with_suffix(".tmp")
            partial_temp.write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            partial_temp.replace(partial_output)
            if case["category"] in {"明确拒绝", "边界"}:
                boundary_hits.append(
                    bool(set(case["gold_chunks"]) & set(retrieved[:20]))
                )
            if index % 10 == 0:
                print(f"progress={index}/{len(cases)}", flush=True)
            print(f"case_done={index}/{len(cases)} id={case['id']}", flush=True)

        with read_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*), count(*) FILTER (
                    WHERE scene_snapshot <> '{}'::jsonb
                      AND retrieval_plan <> '{}'::jsonb
                      AND evidence_assessment <> '{}'::jsonb
                )
                FROM retrieval_traces
                WHERE request_id = ANY(%s)
                """,
                (trace_ids,),
            )
            trace_count, complete_trace_count = cursor.fetchone()

    results = evaluate(cases, predictions)
    results["boundary_key_knowledge_recall"] = (
        sum(boundary_hits) / len(boundary_hits)
    )
    results["source_coverage"] = _source_coverage(args.corpus_version)
    results["retrieval_trace_count"] = trace_count
    results["retrieval_trace_completeness"] = (
        complete_trace_count / trace_count if trace_count else 0
    )
    results["release_gates"] = {
        "gold_evidence_recall_at_20": results["recall_at_20"] >= 0.90,
        "ndcg_at_10": results["ndcg_at_10"] >= 0.75,
        "boundary_key_knowledge_recall": (
            results["boundary_key_knowledge_recall"] == 1
        ),
        "forbidden_chunk_rejection_rate": (
            results["forbidden_chunk_rejection_rate"] >= 0.95
        ),
        "no_evidence_accuracy": results["no_evidence_accuracy"] >= 0.90,
        "source_coverage": results["source_coverage"] == 1,
        "retrieval_trace_completeness": (
            results["retrieval_trace_completeness"] == 1
        ),
    }
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.predictions_output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
