from __future__ import annotations

from typing import Protocol


class RerankerUnavailable(RuntimeError):
    pass


class Reranker(Protocol):
    model_name: str

    def score(self, scene: str, query: str, documents: list[str]) -> list[float]: ...


def rerank(
    candidates: list[dict],
    scene: str,
    query: str,
    model: Reranker,
    limit: int = 6,
) -> list[dict]:
    inputs = [
        f"{candidate.get('title', '')}\n"
        f"{' > '.join(candidate.get('heading_path', []))}\n"
        f"{candidate.get('content', '')}"
        for candidate in candidates[:15]
    ]
    scores = model.score(scene, query, inputs)
    if len(scores) != len(inputs):
        raise RerankerUnavailable("reranker returned an invalid score count")
    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates[:15], scores, strict=True)
    ]
    return sorted(scored, key=lambda item: item["rerank_score"], reverse=True)[:limit]
