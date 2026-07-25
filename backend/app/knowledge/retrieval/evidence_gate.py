from __future__ import annotations

from app.knowledge.domain.models import EvidenceAssessment


OPPOSING_ACTIONS = {
    frozenset(("advance", "stop")),
    frozenset(("increase_pressure", "reduce_pressure")),
    frozenset(("reply_now", "do_not_reply")),
}


def assess_evidence(
    candidates: list[dict],
    required_topics: list[str],
    excluded_topics: list[str],
    reranker_threshold: float | None = None,
) -> EvidenceAssessment:
    required = set(required_topics)
    excluded = set(excluded_topics)
    selected_ids = [str(candidate["chunk_id"]) for candidate in candidates]
    covered = set().union(*(set(candidate.get("topics", [])) for candidate in candidates)) if candidates else set()
    excluded_hits = sorted(covered & excluded)
    missing = sorted(required - covered)
    actions = {
        action
        for candidate in candidates
        for action in candidate.get("action_labels", [])
    }
    conflicting_ids: set[str] = set()
    if any(pair <= actions for pair in OPPOSING_ACTIONS):
        conflicting_ids = {
            str(candidate["chunk_id"])
            for candidate in candidates
            if set(candidate.get("action_labels", [])) & actions
        }

    viable = [
        candidate
        for candidate in candidates
        if not set(candidate.get("topics", [])) & excluded
        and not candidate.get("not_applicable_conflict", False)
    ]
    if not viable or (required and required.isdisjoint(covered)):
        status = "insufficient"
    elif conflicting_ids:
        status = "conflicting"
    elif (
        missing
        or not any(
            candidate.get("knowledge_type") in {"principle", "strategy"}
            for candidate in viable
        )
        or (
            reranker_threshold is not None
            and max(float(candidate.get("rerank_score", 0)) for candidate in viable)
            < reranker_threshold
        )
    ):
        status = "partially_sufficient"
    else:
        status = "sufficient"
    return EvidenceAssessment(
        status=status,
        covered_topics=sorted(covered & required),
        missing_topics=missing,
        excluded_topic_hits=excluded_hits,
        conflicting_chunk_ids=sorted(conflicting_ids),
        selected_chunk_ids=selected_ids,
    )
