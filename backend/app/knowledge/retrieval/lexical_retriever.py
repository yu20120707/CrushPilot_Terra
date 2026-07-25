from __future__ import annotations

from typing import Any

from .tokenizer import build_query_tokens


FILTER_SQL = {
    "review_status": "c.review_status = ANY(%s)",
    "task_types": "COALESCE(c.metadata->'task_types', '[]'::jsonb) ?| %s",
    "relationship_stages": (
        "COALESCE(c.metadata->'relationship_stages', '[]'::jsonb) ?| %s"
    ),
    "knowledge_type": "COALESCE(c.metadata->>'knowledge_type', '') = ANY(%s)",
}


def _or_tsquery(tokens: list[str]) -> str:
    return " | ".join(
        f"'{token.replace(chr(39), chr(39) * 2)}'"
        for token in dict.fromkeys(tokens)
    )


class LexicalRetriever:
    """PostgreSQL FTS adapter for one retrieval query."""

    def __init__(self, connection: object):
        self.connection = connection

    def search(
        self,
        query: str,
        required_topics: list[str],
        excluded_topics: list[str],
        *,
        limit: int = 20,
        hard_filters: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        hard_filters = hard_filters or {}
        unknown = set(hard_filters) - set(FILTER_SQL)
        if unknown:
            raise ValueError(f"unsupported hard filters: {sorted(unknown)}")
        if any(not values for values in hard_filters.values()):
            return []
        tokens = build_query_tokens(query, required_topics)
        if not tokens:
            return []
        filter_sql = "".join(
            f"\n                  AND {FILTER_SQL[key]}"
            for key in sorted(hard_filters)
        )
        filter_params = [hard_filters[key] for key in sorted(hard_filters)]
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT c.id AS chunk_id, c.document_id, c.parent_section_id,
                       c.title, c.heading_path, c.content, c.metadata,
                       ts_rank_cd(c.search_vector, q.query, 2)
                         + CASE
                             WHEN COALESCE(c.metadata->'topics', '[]'::jsonb) ?| %s
                             THEN 0.02
                             ELSE 0
                           END AS score
                FROM knowledge_corpus_versions v
                JOIN knowledge_chunks c ON c.corpus_version = v.version
                CROSS JOIN to_tsquery('simple', %s) AS q(query)
                WHERE v.status = 'published'
                  AND c.review_status = 'approved'
                  AND (
                    c.search_vector @@ q.query
                    OR COALESCE(c.metadata->'topics', '[]'::jsonb) ?| %s
                  )
                  AND NOT (COALESCE(c.metadata->'topics', '[]'::jsonb) ?| %s)
                  {filter_sql}
                ORDER BY score DESC, c.id
                LIMIT %s
                """,
                (
                    required_topics,
                    _or_tsquery(tokens),
                    required_topics,
                    excluded_topics,
                    *filter_params,
                    min(limit, 20),
                ),
            )
            columns = [column.name for column in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for result in results:
                for key in ("chunk_id", "document_id", "parent_section_id"):
                    if result.get(key) is not None:
                        result[key] = str(result[key])
            return results
