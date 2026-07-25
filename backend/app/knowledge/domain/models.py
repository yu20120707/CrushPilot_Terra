from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "counterpart", "assistant"]
    content: str
    created_at: datetime | None = None


class ConfirmedFact(BaseModel):
    fact_id: str
    relationship_id: str | None = None
    content: str
    source_message_ids: list[str]
    confidence: Literal["explicit", "confirmed"]


class ConversationContext(BaseModel):
    current_message: str
    recent_messages: list[ConversationMessage] = Field(max_length=12)
    conversation_summary: str | None = None
    relationship_id: str | None = None
    relationship_facts: list[ConfirmedFact] = Field(default_factory=list, max_length=20)
    user_preferences: list[ConfirmedFact] = Field(default_factory=list, max_length=10)


class SceneSnapshot(BaseModel):
    task_type: Literal[
        "reply",
        "relationship_analysis",
        "invitation",
        "conflict_repair",
        "emotional_support",
        "boundary",
        "relationship_exit",
        "general_advice",
    ]
    recommended_action: Literal[
        "respond",
        "support",
        "advance",
        "observe",
        "reduce_pressure",
        "clarify",
        "repair",
        "stop",
        "exit",
        "safety_exit",
    ]
    relationship_stage: str | None = None
    current_event: str
    user_goal: str
    known_facts: list[str]
    uncertain_inferences: list[str]
    key_unknowns: list[str]
    counterpart_signals: list[str]
    explicit_boundaries: list[str]
    active_skill_scenarios: list[str]
    confidence: float = Field(ge=0, le=1)


class RetrievalPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)
    required_topics: list[str]
    optional_topics: list[str]
    excluded_topics: list[str]
    hard_filters: dict[str, list[str]]
    soft_preferences: dict[str, list[str]]
    lexical_top_k: int = Field(default=20, ge=1)
    vector_top_k: int = Field(default=20, ge=1)
    fusion_top_k: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=15, ge=1)
    final_top_k: int = Field(default=6, ge=1)
    need_parent_context: bool = True
    allow_no_evidence: bool = True


class SceneAndRetrievalPlan(BaseModel):
    scene: SceneSnapshot
    retrieval: RetrievalPlan


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_section_id: str | None = None
    title: str
    heading_path: list[str]
    content: str
    section_summary: str | None = None
    knowledge_type: Literal[
        "hard_rule",
        "principle",
        "strategy",
        "example",
        "counterexample",
        "source_note",
    ]
    topics: list[str]
    task_types: list[str]
    relationship_stages: list[str]
    action_labels: list[str]
    applicable_when: list[str]
    not_applicable_when: list[str]
    evidence_level: Literal["L1", "L2", "L3", "L4"]
    review_status: Literal["draft", "approved", "rejected", "deprecated"]
    priority: int
    source_path: str
    source_sha256: str
    corpus_version: str
    token_count: int = Field(ge=1)


class EvidenceAssessment(BaseModel):
    status: Literal[
        "sufficient",
        "partially_sufficient",
        "conflicting",
        "insufficient",
    ]
    covered_topics: list[str]
    missing_topics: list[str]
    excluded_topic_hits: list[str]
    conflicting_chunk_ids: list[str]
    selected_chunk_ids: list[str]
    fallback_reason: str | None = None


class RetrievalTrace(BaseModel):
    request_id: str
    conversation_id: str
    skill_version: str
    skill_sha256: str
    corpus_version: str
    embedding_model: str | None = None
    reranker_model: str | None = None
    scene_snapshot: dict
    retrieval_plan: dict
    lexical_candidates: list[dict]
    vector_candidates: list[dict]
    fused_candidates: list[dict]
    reranked_candidates: list[dict]
    selected_chunks: list[dict]
    evidence_assessment: dict
    fallback_reason: str | None = None
    latency_ms: dict[str, float]
