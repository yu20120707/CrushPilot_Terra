from __future__ import annotations

import json
import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from ..governance.corpus_version import CorpusReadiness, CorpusVersion


_TOKEN = re.compile(r"[A-Za-z0-9._:/-]{1,128}")
_ID_KEYS = {
    "request_id", "conversation_id", "message_id", "chunk_id", "document_id",
    "source_id",
}
_ID_LIST_KEYS = {"message_ids", "chunk_ids", "selected_chunk_ids", "conflicting_chunk_ids"}
_VERSION_KEYS = {"skill_version", "corpus_version"}
_SCORE_KEYS = {"score", "rrf_score", "rerank_score", "confidence"}
_INTEGER_KEYS = {"rank", "query_index"}
_ENUM_KEYS = {"status", "fallback_reason"}
_LATENCY_KEYS = {
    "scene_analysis_latency_ms",
    "lexical_retrieval_latency_ms",
    "vector_retrieval_latency_ms",
    "rerank_latency_ms",
    "retrieval_failure_latency_ms",
    "generation_latency_ms",
}
_DEBUG_SAMPLE_MAX_LENGTH = 200


def _safe_token(value: Any) -> str | None:
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def _require_token(name: str, value: Any) -> str:
    safe = _safe_token(value)
    if safe is None:
        raise ValueError(f"invalid trace {name}")
    return safe


def _minimize_trace(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    minimized: dict[str, Any] = {}
    for key, item in value.items():
        if key in _ID_KEYS | _VERSION_KEYS | _ENUM_KEYS:
            safe = _safe_token(item)
            if safe is not None:
                minimized[key] = safe
        elif key in _ID_LIST_KEYS and isinstance(item, list):
            safe_items = [safe for entry in item if (safe := _safe_token(entry)) is not None]
            minimized[key] = safe_items
        elif key in _SCORE_KEYS and isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isfinite(item):
                minimized[key] = item
        elif key in _INTEGER_KEYS and isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            minimized[key] = item
    return minimized


def _minimize_latency(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: float(item)
        for key, item in value.items()
        if key in _LATENCY_KEYS
        and isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(item)
        and item >= 0
    }


def _safe_token_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [safe for item in value if (safe := _safe_token(item)) is not None]


def _safe_debug_sample(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > _DEBUG_SAMPLE_MAX_LENGTH:
        return None
    return value if all(character >= " " or character in "\t\n" for character in value) else None


def _minimize_scene(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "task_type_id",
        "recommended_action_id",
        "relationship_stage_id",
        "current_event_id",
        "user_goal_id",
    ):
        safe = _safe_token(value.get(key))
        if safe is not None:
            result[key] = safe
    scenarios = _safe_token_list(value.get("active_skill_scenario_ids"))
    if scenarios:
        result["active_skill_scenario_ids"] = scenarios
    sample = _safe_debug_sample(value.get("current_event_sample"))
    if sample is not None:
        result["current_event_sample"] = sample
    return result


def _minimize_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: _safe_token_list(value.get(key))
        for key in ("query_ids", "required_topic_ids", "excluded_topic_ids")
    }
    hard_filters = value.get("hard_filter_ids")
    if isinstance(hard_filters, dict):
        result["hard_filter_ids"] = {
            key: safe_values
            for key, item in hard_filters.items()
            if (safe_key := _safe_token(key)) is not None
            and (safe_values := _safe_token_list(item))
            for key in [safe_key]
        }
    soft_preferences = value.get("soft_preference_ids")
    if isinstance(soft_preferences, dict):
        result["soft_preference_ids"] = {
            key: safe_values
            for key, item in soft_preferences.items()
            if (safe_key := _safe_token(key)) is not None
            and (safe_values := _safe_token_list(item))
            for key in [safe_key]
        }
    samples = value.get("query_samples")
    if isinstance(samples, list):
        safe_samples = [
            safe for item in samples if (safe := _safe_debug_sample(item)) is not None
        ]
        if safe_samples:
            result["query_samples"] = safe_samples
    return result


def _safe_reason(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    reasons = value.split(",")
    allowed = {
        "vector_unavailable",
        "reranker_unavailable",
        "scene_analysis_failed",
        "retrieval_failed",
        "generation_failed",
    }
    return value if reasons and set(reasons) <= allowed else None


class PostgresKnowledgeRepository:
    def __init__(self, connection: object):
        self.connection = connection

    def create_staging_version(self, corpus: CorpusVersion) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_corpus_versions (
                    version, status, embedding_model, embedding_dimension,
                    embedding_normalization, tokenizer_version, chunker_version
                ) VALUES (%s, 'staging', %s, %s, %s, %s, %s)
                """,
                (
                    corpus.version,
                    corpus.embedding_model,
                    corpus.embedding_dimension,
                    corpus.embedding_normalization,
                    corpus.tokenizer_version,
                    corpus.chunker_version,
                ),
            )
        self.connection.commit()

    def write_staging_build(
        self,
        build: Any,
        search_tokens: dict[str, dict[str, str]],
    ) -> None:
        report = build.report
        corpus = CorpusVersion(
            report.corpus_version,
            report.embedding_model,
            report.embedding_dimension,
            report.embedding_normalization,
            report.tokenizer_version,
            report.chunker_version,
        )
        if set(search_tokens) != {record.chunk.chunk_id for record in build.chunks}:
            raise ValueError("search tokens must cover every chunk exactly")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO knowledge_corpus_versions (
                        version, status, embedding_model, embedding_dimension,
                        embedding_normalization, tokenizer_version, chunker_version
                    ) VALUES (%s, 'staging', %s, %s, %s, %s, %s)
                    """,
                    (
                        corpus.version, corpus.embedding_model, corpus.embedding_dimension,
                        corpus.embedding_normalization, corpus.tokenizer_version,
                        corpus.chunker_version,
                    ),
                )
                documents: dict[str, Any] = {}
                sections: dict[str, tuple[str, str, list[str], str]] = {}
                for record in build.chunks:
                    chunk = record.chunk
                    documents.setdefault(chunk.document_id, chunk)
                    if chunk.parent_section_id:
                        sections.setdefault(
                            chunk.parent_section_id,
                            (
                                chunk.document_id,
                                chunk.heading_path[-2],
                                chunk.heading_path[:-1],
                                chunk.content[:600],
                            ),
                        )
                for document_id, chunk in documents.items():
                    cursor.execute(
                        """
                        INSERT INTO knowledge_documents
                            (id, source_path, title, source_sha256, metadata, corpus_version)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            document_id, chunk.source_path, chunk.heading_path[0],
                            chunk.source_sha256, json.dumps({}), corpus.version,
                        ),
                    )
                for ordinal, (
                    section_id,
                    (document_id, title, path, summary),
                ) in enumerate(sections.items()):
                    cursor.execute(
                        """
                        INSERT INTO knowledge_sections
                            (id, document_id, parent_section_id, title, heading_path,
                             summary, ordinal, corpus_version)
                        VALUES (%s, %s, NULL, %s, %s::jsonb, %s, %s, %s)
                        """,
                        (
                            section_id,
                            document_id,
                            title,
                            json.dumps(path),
                            summary,
                            ordinal,
                            corpus.version,
                        ),
                    )
                for record in build.chunks:
                    chunk = record.chunk
                    weighted_tokens = search_tokens[chunk.chunk_id]
                    if set(weighted_tokens) != {"A", "B", "C", "D"} or not all(
                        isinstance(value, str) for value in weighted_tokens.values()
                    ):
                        raise ValueError("search tokens require application-tokenized A/B/C/D fields")
                    metadata = {
                        "topics": chunk.topics,
                        "task_types": chunk.task_types,
                        "relationship_stages": chunk.relationship_stages,
                        "action_labels": chunk.action_labels,
                        "applicable_when": chunk.applicable_when,
                        "not_applicable_when": chunk.not_applicable_when,
                        "knowledge_type": chunk.knowledge_type,
                        "evidence_level": chunk.evidence_level,
                        "priority": chunk.priority,
                        "token_count": chunk.token_count,
                        "previous_chunk_id": record.relationships.previous_chunk_id,
                        "next_chunk_id": record.relationships.next_chunk_id,
                    }
                    cursor.execute(
                        """
                        INSERT INTO knowledge_chunks (
                            id, document_id, parent_section_id, title, heading_path,
                            content, metadata, search_tokens, search_vector, review_status,
                            source_sha256, corpus_version
                        ) VALUES (
                            %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s,
                            setweight(to_tsvector('simple', %s), 'A') ||
                            setweight(to_tsvector('simple', %s), 'B') ||
                            setweight(to_tsvector('simple', %s), 'C') ||
                            setweight(to_tsvector('simple', %s), 'D'),
                            %s, %s, %s
                        )
                        """,
                        (
                            chunk.chunk_id, chunk.document_id, chunk.parent_section_id,
                            chunk.title, json.dumps(chunk.heading_path), chunk.content,
                            json.dumps(metadata), " ".join(weighted_tokens.values()),
                            weighted_tokens["A"], weighted_tokens["B"],
                            weighted_tokens["C"], weighted_tokens["D"],
                            chunk.review_status, chunk.source_sha256, corpus.version,
                        ),
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def load_expansion_context(
        self, selected: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        parent_ids = sorted(
            {
                str(item["parent_section_id"])
                for item in selected
                if item.get("parent_section_id")
            }
        )
        neighbor_ids = sorted(
            {
                str(item[key])
                for item in selected
                for key in ("previous_chunk_id", "next_chunk_id")
                if item.get(key)
            }
        )
        context: dict[str, dict[str, Any]] = {}
        with self.connection.cursor() as cursor:
            if parent_ids:
                cursor.execute(
                    """
                    SELECT s.id, s.document_id, COALESCE(s.summary, s.title)
                    FROM knowledge_sections s
                    JOIN knowledge_corpus_versions v
                      ON v.version = s.corpus_version
                    WHERE v.status = 'published' AND s.id = ANY(%s::uuid[])
                    """,
                    (parent_ids,),
                )
                for section_id, document_id, content in cursor.fetchall():
                    context[str(section_id)] = {
                        "chunk_id": str(section_id),
                        "document_id": str(document_id),
                        "content": content,
                        "token_count": len(str(content)),
                        "context_kind": "parent",
                    }
            if neighbor_ids:
                cursor.execute(
                    """
                    SELECT c.id, c.document_id, c.content, c.metadata
                    FROM knowledge_chunks c
                    JOIN knowledge_corpus_versions v
                      ON v.version = c.corpus_version
                    WHERE v.status = 'published'
                      AND c.review_status = 'approved'
                      AND c.id = ANY(%s::uuid[])
                    """,
                    (neighbor_ids,),
                )
                for chunk_id, document_id, content, metadata in cursor.fetchall():
                    metadata = metadata if isinstance(metadata, dict) else {}
                    context[str(chunk_id)] = {
                        "chunk_id": str(chunk_id),
                        "document_id": str(document_id),
                        "content": content,
                        "token_count": int(metadata.get("token_count") or len(content)),
                        "context_kind": "neighbor",
                    }
        return context

    def write_embeddings(
        self,
        corpus_version: str,
        embedding_model: str,
        embeddings: dict[str, Sequence[float]],
    ) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT embedding_model, embedding_dimension
                    FROM knowledge_corpus_versions
                    WHERE version = %s AND status = 'staging'
                    FOR UPDATE
                    """,
                    (corpus_version,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("staging corpus version not found")
                if row[0] != embedding_model:
                    raise ValueError("embedding model mismatch")
                dimension = row[1]
                for chunk_id, embedding in embeddings.items():
                    if len(embedding) != dimension or not all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        for value in embedding
                    ):
                        raise ValueError("embedding dimension or values are invalid")
                    vector = "[" + ",".join(str(value) for value in embedding) + "]"
                    cursor.execute(
                        """
                        INSERT INTO knowledge_embeddings (
                            chunk_id, embedding_model, embedding_dimension,
                            embedding, corpus_version
                        ) VALUES (%s, %s, %s, %s::vector, %s)
                        """,
                        (chunk_id, embedding_model, dimension, vector, corpus_version),
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def exact_cosine_search(
        self,
        embedding: Sequence[float],
        embedding_model: str,
        limit: int = 20,
        hard_filters: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        allowed_filters = {
            "review_status", "task_types", "relationship_stages", "knowledge_type"
        }
        hard_filters = hard_filters or {}
        if unknown := set(hard_filters) - allowed_filters:
            raise ValueError(f"unknown hard filters: {sorted(unknown)}")
        for key, values in hard_filters.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"hard filter {key} must be a non-empty list")
            if any(_safe_token(value) is None for value in values):
                raise ValueError(f"hard filter {key} contains an invalid value")
        review_status = hard_filters.get("review_status")
        task_types = hard_filters.get("task_types")
        relationship_stages = hard_filters.get("relationship_stages")
        knowledge_type = hard_filters.get("knowledge_type")
        vector = "[" + ",".join(str(value) for value in embedding) + "]"
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id AS chunk_id, c.document_id, c.parent_section_id,
                       c.title, c.heading_path, c.content, c.metadata,
                       1 - (e.embedding <=> %s::vector) AS score
                FROM knowledge_corpus_versions v
                JOIN knowledge_chunks c ON c.corpus_version = v.version
                JOIN knowledge_embeddings e
                  ON e.chunk_id = c.id
                 AND e.corpus_version = c.corpus_version
                 AND e.corpus_version = v.version
                WHERE v.status = 'published'
                  AND c.review_status = 'approved'
                  AND e.embedding_model = %s
                  AND e.embedding_dimension = %s
                  AND (%s::text[] IS NULL OR c.review_status = ANY(%s::text[]))
                  AND (%s::text[] IS NULL OR c.metadata -> 'task_types' ?| %s::text[])
                  AND (%s::text[] IS NULL OR c.metadata -> 'relationship_stages' ?| %s::text[])
                  AND (%s::text[] IS NULL OR c.metadata ->> 'knowledge_type' = ANY(%s::text[]))
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vector, embedding_model, len(embedding),
                    review_status, review_status, task_types, task_types,
                    relationship_stages, relationship_stages,
                    knowledge_type, knowledge_type, vector, limit,
                ),
            )
            columns = [column.name for column in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            for result in results:
                for key in ("chunk_id", "document_id", "parent_section_id"):
                    if result.get(key) is not None:
                        result[key] = str(result[key])
            return results

    def readiness(
        self,
        expected: CorpusVersion,
        allow_lexical_only: bool = False,
    ) -> CorpusReadiness:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT version, embedding_model, embedding_dimension,
                       embedding_normalization, tokenizer_version, chunker_version,
                       (SELECT count(*) FROM knowledge_chunks c
                        WHERE c.corpus_version = v.version
                          AND c.review_status = 'approved') AS chunk_count,
                       (SELECT count(*) FROM knowledge_embeddings e
                        JOIN knowledge_chunks c
                          ON c.id = e.chunk_id
                         AND c.corpus_version = e.corpus_version
                        WHERE e.corpus_version = v.version
                          AND c.review_status = 'approved'
                          AND e.embedding_model = v.embedding_model
                          AND e.embedding_dimension = v.embedding_dimension) AS embedding_count
                FROM knowledge_corpus_versions v
                WHERE status = 'published'
                """
            )
            row = cursor.fetchone()
        if row is None:
            return CorpusReadiness(False, reason="no published corpus")
        version, model, dimension, normalization, tokenizer, chunker, count, embedding_count = row
        if version != expected.version or tokenizer != expected.tokenizer_version or chunker != expected.chunker_version:
            return CorpusReadiness(False, reason="corpus version mismatch")
        if count < 1:
            return CorpusReadiness(False, reason="published corpus is empty")
        model_matches = (
            model == expected.embedding_model
            and dimension == expected.embedding_dimension
            and normalization == expected.embedding_normalization
        )
        if model_matches and embedding_count == count:
            return CorpusReadiness(True)
        if allow_lexical_only:
            reason = (
                "published corpus embeddings are incomplete"
                if model_matches
                else "embedding metadata mismatch"
            )
            return CorpusReadiness(True, lexical_only=True, reason=reason)
        reason = (
            "published corpus embeddings are incomplete"
            if model_matches
            else "embedding metadata mismatch"
        )
        return CorpusReadiness(False, reason=reason)

    def write_trace(self, trace: dict[str, Any]) -> None:
        try:
            self._write_trace(trace)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def mark_trace_failure(
        self, request_id: str, reason: str, latency_ms: float
    ) -> None:
        safe_id = _require_token("request_id", request_id)
        safe_reason = _safe_reason(reason)
        if safe_reason is None or not isinstance(latency_ms, (int, float)):
            raise ValueError("invalid trace failure update")
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE retrieval_traces
                    SET fallback_reason = %s,
                        latency_ms = latency_ms ||
                          jsonb_build_object('generation_latency_ms', %s::double precision)
                    WHERE request_id = %s
                    """,
                    (safe_reason, float(latency_ms), safe_id),
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _write_trace(self, trace: dict[str, Any]) -> None:
        candidate_stages = (
            "lexical_candidates",
            "vector_candidates",
            "fused_candidates",
            "reranked_candidates",
            "selected_chunks",
        )
        request_id = _require_token("request_id", trace.get("request_id"))
        conversation_id = _require_token("conversation_id", trace.get("conversation_id"))
        skill_version = _require_token("skill_version", trace.get("skill_version"))
        skill_sha256 = _require_token("skill_sha256", trace.get("skill_sha256"))
        corpus_version = _require_token("corpus_version", trace.get("corpus_version"))
        embedding_model = (
            _require_token("embedding_model", trace["embedding_model"])
            if trace.get("embedding_model") is not None
            else None
        )
        reranker_model = (
            _require_token("reranker_model", trace["reranker_model"])
            if trace.get("reranker_model") is not None
            else None
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO retrieval_traces (
                    request_id, conversation_id, skill_version, skill_sha256,
                    corpus_version, embedding_model, reranker_model, scene_snapshot,
                    retrieval_plan, evidence_assessment, fallback_reason, latency_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                          %s::jsonb, %s, %s::jsonb)
                """,
                (
                    request_id, conversation_id, skill_version,
                    skill_sha256, corpus_version, embedding_model, reranker_model,
                    json.dumps(_minimize_scene(trace["scene_snapshot"])),
                    json.dumps(_minimize_plan(trace["retrieval_plan"])),
                    json.dumps(_minimize_trace(trace["evidence_assessment"])),
                    _safe_reason(trace.get("fallback_reason")),
                    json.dumps(_minimize_latency(trace["latency_ms"])),
                ),
            )
            for stage in candidate_stages:
                candidates = trace.get(stage)
                if not isinstance(candidates, list):
                    continue
                for rank, candidate in enumerate(candidates, 1):
                    minimized = _minimize_trace(candidate)
                    score = minimized.get("score")
                    if score is None and stage == "fused_candidates":
                        score = minimized.get("rrf_score")
                    if score is None and stage == "reranked_candidates":
                        score = minimized.get("rerank_score")
                    cursor.execute(
                        """
                        INSERT INTO retrieval_trace_candidates
                            (request_id, stage, chunk_id, rank, score, details)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            request_id, stage, minimized.get("chunk_id"),
                            rank, score, json.dumps(minimized),
                        ),
                    )
