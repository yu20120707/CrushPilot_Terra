from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class VectorUnavailable(RuntimeError):
    pass


class VectorRetriever:
    def __init__(
        self,
        repository: object,
        embed: Callable[[str], Sequence[float]],
        embedding_model: str,
    ):
        self.repository = repository
        self.embed = embed
        self.embedding_model = embedding_model

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        hard_filters: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            embedding = self.embed(query)
            kwargs = {"limit": min(limit, 20)}
            if hard_filters:
                kwargs["hard_filters"] = hard_filters
            return self.repository.exact_cosine_search(
                embedding, self.embedding_model, **kwargs
            )
        except Exception as exc:
            raise VectorUnavailable(str(exc)) from exc
