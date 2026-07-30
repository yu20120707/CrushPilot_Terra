from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from app.knowledge.domain.models import EvidenceAssessment, RetrievalTrace
from app.knowledge.retrieval.service import RetrievalResult


class LocalJsonRetrievalService:
    """Small local vector-card retriever for development without PostgreSQL."""

    def __init__(self, path: Path) -> None:
        source = json.loads(path.read_text(encoding="utf-8"))
        self.cards = source["cards"]
        self.embedding_model = source["embedding_model"]
        self._encoder: Any | None = None
        self._traces: dict[str, dict[str, Any]] = {}

    def retrieve(self, *, request_id: str, conversation_id: str, skill_version: str,
                 skill_sha256: str, corpus_version: str, scene: Any, plan: Any,
                 embedding_model: str | None = None) -> RetrievalResult:
        started = perf_counter()
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model)
        query = "\n".join(plan.queries)
        query_vector = self._encoder.encode(query, normalize_embeddings=True).tolist()
        candidates = sorted(
            [
                {
                    "chunk_id": card["id"],
                    "document_id": "local-knowledge-vectors",
                    "title": card["id"],
                    "heading_path": ["本地知识卡"],
                    "content": card["text"],
                    "knowledge_type": "strategy",
                    "topics": plan.required_topics,
                    "action_labels": [],
                    "not_applicable_when": [],
                    "score": sum(a * b for a, b in zip(query_vector, card["vector"])),
                }
                for card in self.cards
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[: min(plan.final_top_k, 3)]
        selected = [{**item, "rerank_score": item["score"], "expanded_context": []} for item in candidates]
        assessment = EvidenceAssessment(
            status="sufficient" if selected else "insufficient",
            covered_topics=plan.required_topics if selected else [],
            missing_topics=[] if selected else plan.required_topics,
            excluded_topic_hits=[],
            conflicting_chunk_ids=[],
            selected_chunk_ids=[item["chunk_id"] for item in selected],
            fallback_reason="local_json_vector",
        )
        latency = {"local_vector_retrieval_latency_ms": (perf_counter() - started) * 1000}
        trace = RetrievalTrace(
            request_id=request_id,
            conversation_id=conversation_id,
            skill_version=skill_version,
            skill_sha256=skill_sha256,
            corpus_version=corpus_version,
            embedding_model=self.embedding_model,
            reranker_model=None,
            scene_snapshot={},
            retrieval_plan={"queries": plan.queries, "required_topics": plan.required_topics},
            lexical_candidates=[],
            vector_candidates=[{"chunk_id": item["chunk_id"], "score": item["score"]} for item in candidates],
            fused_candidates=[],
            reranked_candidates=[],
            selected_chunks=[{"chunk_id": item["chunk_id"], "score": item["score"]} for item in selected],
            evidence_assessment=assessment.model_dump(),
            fallback_reason="local_json_vector",
            latency_ms=latency,
        )
        self._traces[conversation_id] = {
            "query": query,
            "selected_chunks": [
                {"chunk_id": item["chunk_id"], "score": round(item["score"], 4), "content": item["content"]}
                for item in selected
            ],
            "fallback_reason": "local_json_vector",
        }
        return RetrievalResult(selected, assessment, trace)

    def trace_for(self, conversation_id: str) -> dict[str, Any] | None:
        return self._traces.get(conversation_id)
