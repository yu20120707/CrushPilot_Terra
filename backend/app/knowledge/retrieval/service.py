from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from time import perf_counter
from typing import Any

from app.knowledge.domain.models import (
    EvidenceAssessment,
    RetrievalPlan,
    RetrievalTrace,
    SceneSnapshot,
    topic_search_text,
)
from app.knowledge.ingestion.chunker import token_count
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.observability.retrieval_trace import controlled_debug_sample

from .diversity import select_diverse
from .evidence_gate import assess_evidence
from .expander import expand_selected
from .fusion import query_weights, weighted_rrf
from .metadata_enricher import isolate_bad_candidates
from .reranker import RerankerUnavailable, rerank
from .vector_retriever import VectorUnavailable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[dict[str, Any]]
    assessment: EvidenceAssessment
    trace: RetrievalTrace


class RetrievalService:
    def __init__(
        self,
        lexical_retriever: object,
        vector_retriever: object,
        reranker: object,
        *,
        trace_writer: object | None = None,
        metrics: RetrievalMetrics | None = None,
        context_loader: object | None = None,
        evidence_budget: int = 3500,
        reranker_threshold: float | None = None,
        trace_debug: bool = False,
    ):
        self.lexical = lexical_retriever
        self.vector = vector_retriever
        self.reranker = reranker
        self.trace_writer = trace_writer
        self.metrics = metrics
        self.context_loader = context_loader
        self.evidence_budget = evidence_budget
        self.reranker_threshold = reranker_threshold
        self.trace_debug = trace_debug

    def retrieve(
        self,
        *,
        request_id: str,
        conversation_id: str,
        skill_version: str,
        skill_sha256: str,
        corpus_version: str,
        scene: SceneSnapshot,
        plan: RetrievalPlan,
        embedding_model: str | None = None,
    ) -> RetrievalResult:
        latency: dict[str, float] = {}
        lexical: list[dict[str, Any]] = []
        vector: list[dict[str, Any]] = []
        fallback_reasons: list[str] = []

        started = perf_counter()
        for query in plan.queries:
            lexical.append(
                self.lexical.search(
                    query,
                    plan.required_topics,
                    plan.excluded_topics,
                    limit=min(plan.lexical_top_k, 20),
                    hard_filters=plan.hard_filters,
                )[: min(plan.lexical_top_k, 20)]
            )
        latency["lexical_retrieval_latency_ms"] = _elapsed_ms(started)

        started = perf_counter()
        try:
            for query in plan.queries:
                vector.append(
                    self.vector.search(
                        f"{query}\n{topic_search_text(plan.required_topics)}".strip(),
                        limit=min(plan.vector_top_k, 20),
                        hard_filters=plan.hard_filters,
                    )[
                        : min(plan.vector_top_k, 20)
                    ]
                )
        except VectorUnavailable:
            vector = [[] for _ in plan.queries]
            fallback_reasons.append("vector_unavailable")
        latency["vector_retrieval_latency_ms"] = _elapsed_ms(started)

        lexical = [isolate_bad_candidates(items) for items in lexical]
        vector = [isolate_bad_candidates(items) for items in vector]
        started = perf_counter()
        candidates = {
            item["chunk_id"]: item
            for results in lexical + vector
            for item in results
        }
        scores = weighted_rrf(
            _ranks(lexical), _ranks(vector), query_weights(len(plan.queries))
        )
        fused = [
            {**candidates[chunk_id], "rrf_score": score}
            for chunk_id, score in sorted(
                scores.items(),
                key=lambda pair: (
                    -pair[1],
                    -_soft_preference_score(candidates[pair[0]], plan.soft_preferences),
                    pair[0],
                ),
            )[: min(plan.fusion_top_k, 20)]
        ]
        latency["fusion_latency_ms"] = _elapsed_ms(started)

        started = perf_counter()
        if fused:
            try:
                reranked = rerank(
                    fused[: min(plan.rerank_top_k, 15)],
                    _scene_text(scene),
                    plan.queries[0],
                    self.reranker,
                    limit=min(plan.rerank_top_k, 15),
                )
            except Exception as exc:
                logger.warning("reranker unavailable: %s", exc)
                reranked = fused[: min(plan.rerank_top_k, 15)]
                fallback_reasons.append("reranker_unavailable")
        else:
            reranked = []
        latency["rerank_latency_ms"] = _elapsed_ms(started)

        started = perf_counter()
        active_scenarios = set(scene.active_skill_scenarios)
        reranked = [
            {
                **candidate,
                "not_applicable_conflict": bool(
                    set(candidate.get("not_applicable_when", []))
                    & active_scenarios
                ),
            }
            for candidate in reranked
        ]
        selected = select_diverse(
            reranked,
            plan.required_topics,
            plan.excluded_topics,
            limit=min(plan.final_top_k, 6),
        )
        latency["diversity_latency_ms"] = _elapsed_ms(started)
        started = perf_counter()
        context_by_id = (
            self.context_loader.load_expansion_context(selected)
            if self.context_loader is not None and selected
            else {}
        )
        expanded = (
            expand_selected(
                selected,
                context_by_id,
                include_parent=plan.need_parent_context,
            )
            if context_by_id
            else [{**candidate, "expanded_context": []} for candidate in selected]
        )
        budgeted = _fit_budget(
            expanded, self.evidence_budget, plan.required_topics
        )
        latency["expansion_latency_ms"] = _elapsed_ms(started)
        started = perf_counter()
        assessment = assess_evidence(
            budgeted,
            plan.required_topics,
            plan.excluded_topics,
            reranker_threshold=self.reranker_threshold,
        )
        if assessment.status == "insufficient":
            budgeted = []
            assessment.selected_chunk_ids = []
        fallback_reason = ",".join(fallback_reasons) or None
        assessment.fallback_reason = fallback_reason
        latency["evidence_gate_latency_ms"] = _elapsed_ms(started)
        trace = RetrievalTrace(
            request_id=request_id,
            conversation_id=conversation_id,
            skill_version=skill_version,
            skill_sha256=skill_sha256,
            corpus_version=corpus_version,
            embedding_model=embedding_model,
            reranker_model=getattr(self.reranker, "model_name", None),
            scene_snapshot=_trace_scene(scene, debug=self.trace_debug),
            retrieval_plan=_trace_plan(plan, debug=self.trace_debug),
            lexical_candidates=_trace_candidates(_flatten(lexical), "score"),
            vector_candidates=_trace_candidates(_flatten(vector), "score"),
            fused_candidates=_trace_candidates(fused, "rrf_score"),
            reranked_candidates=_trace_candidates(reranked, "rerank_score"),
            selected_chunks=_trace_candidates(budgeted, "rerank_score"),
            evidence_assessment=assessment.model_dump(),
            fallback_reason=fallback_reason,
            latency_ms=latency,
        )
        self._record_metrics(latency, assessment.status, fallback_reasons, len(budgeted))
        if self.trace_writer is not None:
            try:
                self.trace_writer.write_trace(trace.model_dump())
            except Exception as exc:
                logger.warning("retrieval trace write failed: %s", exc, exc_info=True)
                if self.metrics is not None:
                    try:
                        self.metrics.observe("trace_write_failure_count", 1)
                    except KeyError:
                        pass
        return RetrievalResult(budgeted, assessment, trace)

    def _record_metrics(
        self,
        latency: dict[str, float],
        status: str,
        fallback_reasons: list[str],
        selected_count: int,
    ) -> None:
        if self.metrics is None:
            return
        for name in (
            "lexical_retrieval_latency_ms",
            "vector_retrieval_latency_ms",
            "rerank_latency_ms",
        ):
            self.metrics.observe(name, latency[name])
        self.metrics.observe("retrieval_no_evidence_rate", status == "insufficient")
        self.metrics.observe("retrieval_partial_evidence_rate", status == "partially_sufficient")
        self.metrics.observe("retrieval_conflict_rate", status == "conflicting")
        self.metrics.observe("vector_fallback_rate", "vector_unavailable" in fallback_reasons)
        self.metrics.observe("reranker_fallback_rate", "reranker_unavailable" in fallback_reasons)
        self.metrics.observe("selected_chunk_count", selected_count)


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def _soft_preference_score(candidate: dict, preferences: dict[str, list[str]]) -> int:
    return sum(
        bool(
            set(
                candidate.get(field, [])
                if isinstance(candidate.get(field, []), list)
                else [candidate.get(field)]
            )
            & set(values)
        )
        for field, values in preferences.items()
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trace_scene(scene: SceneSnapshot, *, debug: bool = False) -> dict[str, Any]:
    snapshot = {
        "task_type_id": scene.task_type,
        "recommended_action_id": scene.recommended_action,
        "relationship_stage_id": scene.relationship_stage,
        "current_event_id": _hash(scene.current_event),
        "user_goal_id": _hash(scene.user_goal),
        "active_skill_scenario_ids": scene.active_skill_scenarios,
    }
    if debug:
        snapshot["current_event_sample"] = controlled_debug_sample(scene.current_event)
    return snapshot


def _trace_plan(plan: RetrievalPlan, *, debug: bool = False) -> dict[str, Any]:
    snapshot = {
        "query_ids": [_hash(query) for query in plan.queries],
        "required_topic_ids": plan.required_topics,
        "excluded_topic_ids": plan.excluded_topics,
        "hard_filter_ids": plan.hard_filters,
        "soft_preference_ids": plan.soft_preferences,
    }
    if debug:
        snapshot["query_samples"] = [
            controlled_debug_sample(query) for query in plan.queries
        ]
    return snapshot


def _trace_candidates(candidates: list[dict], original_score: str) -> list[dict]:
    safe = []
    for candidate in candidates:
        item = {
            key: candidate[key]
            for key in (
                "chunk_id",
                "document_id",
                "query_index",
                "rank",
                "topics",
                "knowledge_type",
                "action_labels",
            )
            if key in candidate
        }
        item["score"] = candidate.get(
            original_score,
            candidate.get("rerank_score", candidate.get("rrf_score", candidate.get("score"))),
        )
        if original_score != "score" and original_score in candidate:
            item[original_score] = candidate[original_score]
        if "content" in candidate:
            item["content_id"] = _hash(candidate["content"])
        safe.append(item)
    return safe


def _ranks(results: list[list[dict[str, Any]]]) -> dict[tuple[str, int], int]:
    return {
        (item["chunk_id"], query_index): rank
        for query_index, items in enumerate(results)
        for rank, item in enumerate(items, 1)
    }


def _flatten(results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {**item, "query_index": query_index, "rank": rank}
        for query_index, items in enumerate(results)
        for rank, item in enumerate(items, 1)
    ]


def _scene_text(scene: SceneSnapshot) -> str:
    return "\n".join(
        (scene.current_event, scene.user_goal, *scene.explicit_boundaries)
    )


def _fit_budget(
    items: list[dict[str, Any]], budget: int, required_topics: list[str]
) -> list[dict[str, Any]]:
    required = set(required_topics)
    score = lambda item: float(
        item.get("rerank_score", item.get("rrf_score", item.get("score", 0)))
    )
    core = sorted(
        (item for item in items if item.get("knowledge_type") != "example"),
        key=lambda item: (
            not bool(set(item.get("topics", [])) & required),
            -score(item),
            str(item["chunk_id"]),
        ),
    )
    examples = sorted(
        (item for item in items if item.get("knowledge_type") == "example"),
        key=lambda item: (-score(item), str(item["chunk_id"])),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for item in core:
        core_cost = int(item.get("token_count") or token_count(item["content"]))
        if used + core_cost > budget:
            continue
        kept = dict(item)
        kept["expanded_context"] = []
        used += core_cost
        selected.append(kept)

    missing = required - {
        topic for item in selected for topic in item.get("topics", [])
    }
    required_examples = [
        item for item in examples if set(item.get("topics", [])) & missing
    ]
    remaining_examples = [
        item for item in examples if item not in required_examples
    ]
    for item in required_examples:
        cost = int(item.get("token_count") or token_count(item["content"]))
        if used + cost <= budget:
            selected.append({**item, "expanded_context": []})
            used += cost

    selected_by_id = {item["chunk_id"]: item for item in selected}
    for context_kind in ("parent", "neighbor"):
        for item in core:
            kept = selected_by_id.get(item["chunk_id"])
            if kept is None:
                continue
            for related in item.get("expanded_context", []):
                if related.get("context_kind") != context_kind:
                    continue
                cost = int(
                    related.get("token_count") or token_count(related["content"])
                )
                if used + cost <= budget:
                    kept["expanded_context"].append(related)
                    used += cost

    for item in remaining_examples:
        cost = int(item.get("token_count") or token_count(item["content"]))
        if used + cost <= budget:
            kept = {**item, "expanded_context": []}
            used += cost
            for related in item.get("expanded_context", []):
                if related.get("context_kind") != "parent":
                    continue
                context_cost = int(
                    related.get("token_count") or token_count(related["content"])
                )
                if used + context_cost <= budget:
                    kept["expanded_context"].append(related)
                    used += context_cost
            selected.append(kept)
    return selected
