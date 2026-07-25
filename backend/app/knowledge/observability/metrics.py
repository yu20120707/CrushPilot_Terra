from __future__ import annotations

from collections import defaultdict
from threading import Lock


REQUIRED_METRICS = (
    "scene_analysis_latency_ms",
    "lexical_retrieval_latency_ms",
    "vector_retrieval_latency_ms",
    "rerank_latency_ms",
    "retrieval_failure_latency_ms",
    "generation_latency_ms",
    "retrieval_no_evidence_rate",
    "retrieval_partial_evidence_rate",
    "retrieval_conflict_rate",
    "vector_fallback_rate",
    "reranker_fallback_rate",
    "selected_chunk_count",
    "corpus_version_mismatch_count",
    "trace_write_failure_count",
)


class RetrievalMetrics:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def observe(self, name: str, value: float) -> None:
        if name not in REQUIRED_METRICS:
            raise KeyError(f"unknown retrieval metric: {name}")
        with self._lock:
            self._values[name].append(float(value))

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                name: {
                    "count": float(len(values)),
                    "sum": sum(values),
                    "latest": values[-1] if values else 0.0,
                }
                for name in REQUIRED_METRICS
                for values in [self._values.get(name, [])]
            }
