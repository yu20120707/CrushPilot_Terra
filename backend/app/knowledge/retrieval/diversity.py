from __future__ import annotations

from .tokenizer import tokenize


def _overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def select_diverse(
    ranked: list[dict],
    required_topics: list[str],
    excluded_topics: list[str],
    limit: int = 6,
    per_document: int = 2,
) -> list[dict]:
    excluded = set(excluded_topics)
    required = set(required_topics)
    eligible = [
        candidate
        for candidate in ranked
        if not set(candidate.get("topics", [])) & excluded
        and not candidate.get("not_applicable_conflict", False)
    ]
    selected: list[dict] = []
    seen: set[str] = set()
    covered: set[str] = set()
    document_counts: dict[str, int] = {}
    while eligible and len(selected) < limit:
        eligible.sort(
            key=lambda candidate: (
                -len((set(candidate.get("topics", [])) & required) - covered),
                candidate.get("knowledge_type") == "example",
                -float(candidate.get("rerank_score", candidate.get("rrf_score", 0))),
            )
        )
        candidate = eligible.pop(0)
        chunk_id = str(candidate["chunk_id"])
        document_id = str(candidate["document_id"])
        if chunk_id in seen or document_counts.get(document_id, 0) >= per_document:
            continue
        if any(
            _overlap(str(candidate.get("content", "")), str(existing.get("content", "")))
            >= 0.8
            for existing in selected
        ):
            continue
        selected.append(candidate)
        seen.add(chunk_id)
        covered.update(set(candidate.get("topics", [])) & required)
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
    return selected
