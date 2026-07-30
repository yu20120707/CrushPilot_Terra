from __future__ import annotations


def weighted_rrf(
    lexical_ranks: dict[tuple[str, int], int],
    vector_ranks: dict[tuple[str, int], int],
    query_weights: list[float],
    k: int = 60,
    lexical_weight: float = 0.45,
    vector_weight: float = 0.55,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for query_index, query_weight in enumerate(query_weights):
        chunk_ids = {
            chunk_id for chunk_id, index in lexical_ranks if index == query_index
        } | {
            chunk_id for chunk_id, index in vector_ranks if index == query_index
        }
        for chunk_id in chunk_ids:
            score = 0.0
            lexical_rank = lexical_ranks.get((chunk_id, query_index))
            vector_rank = vector_ranks.get((chunk_id, query_index))
            if lexical_rank is not None:
                score += lexical_weight / (k + lexical_rank)
            if vector_rank is not None:
                score += vector_weight / (k + vector_rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + query_weight * score
    return scores


def query_weights(query_count: int) -> list[float]:
    return [1.0] + [0.8] * max(query_count - 1, 0)
