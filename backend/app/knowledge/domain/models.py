from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


TOPIC_SEARCH_TERMS = {
    "rejection": ("拒绝", "婉拒", "不见面"),
    "boundary": ("边界", "同意", "越界", "施压"),
    "reduce_pressure": ("降压", "留空间", "不施压", "降低压力", "只回", "不追问", "收尾"),
    "emotional_support": ("情绪", "安慰", "倾听", "共情", "情绪价值"),
    "low_pressure_communication": ("松弛", "低压力", "自然沟通"),
    "invitation": ("邀约", "约会", "见面"),
    "conflict_repair": ("冲突", "吵架", "修复", "道歉"),
    "relationship_exit": ("分手", "退出", "背叛"),
    "reciprocity": ("互惠", "投入", "失衡"),
    "digital_context": ("延迟回复", "晚回", "文字沟通", "数字边界", "线上互动"),
    "manipulation": ("操控", "pua", "欺骗", "强迫"),
}
TOPIC_PLANNING_DESCRIPTIONS = {
    "rejection": "明确或含蓄表达不接受邀约、推进或关系请求",
    "boundary": "对方的意愿、同意、拒绝、改变主意及需要尊重的行为边界",
    "reduce_pressure": "冷淡、忙碌、未及时回复或信号不明时减少追问并留出空间",
    "emotional_support": "难过、焦虑、疲惫、受挫等情绪需要倾听、安慰或共情",
    "low_pressure_communication": "日常接话、自然回复和不施压的轻松沟通",
    "invitation": "提出、接受、拒绝或调整具体见面和约会安排",
    "conflict_repair": "争执、误会、伤害后的道歉、协商和关系修复",
    "relationship_exit": "分手、退出、背叛或结束关系",
    "reciprocity": "双方主动、投入、回应、关系期待或继续了解是否对等",
    "digital_context": "消息延迟、已读、点赞、表情、深夜集中回复等线上互动语境",
    "manipulation": "操控、欺骗、强迫或其他不安全的关系策略",
}
PLANNING_ONLY_TOPICS = frozenset({
    "aggressive_pursuit",
    "conservative_action",
    "forced_disclosure",
    "mind_reading",
    "relationship_escalation",
    "uncertainty",
})
CONTROLLED_TOPICS = frozenset(TOPIC_SEARCH_TERMS) | PLANNING_ONLY_TOPICS | {
    "general_relationship_advice"
}


def topic_search_text(topics: list[str]) -> str:
    return " ".join(
        term for topic in topics for term in TOPIC_SEARCH_TERMS.get(topic, (topic,))
    )


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
    ] = Field(description="reply=代写接话；relationship_analysis=判断信号或趋势；其余按场景名选择")
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
    ] = Field(description="明确拒绝或边界选 stop；冷淡或延迟选 reduce_pressure；模糊信号选 observe")
    relationship_stage: str | None = None
    current_event: str
    user_goal: str
    known_facts: list[str]
    uncertain_inferences: list[str]
    key_unknowns: list[str]
    counterpart_signals: list[str]
    explicit_boundaries: list[str]
    active_skill_scenarios: list[str] = Field(
        description="仅在明确命中时选择 explicit_rejection、emotional_support 或 insufficient_information"
    )
    confidence: float = Field(ge=0, le=1)


class RetrievalPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=3)
    required_topics: list[str] = Field(
        description=f"只允许受控 topic id：{', '.join(sorted(CONTROLLED_TOPICS))}"
    )
    optional_topics: list[str]
    excluded_topics: list[str] = Field(
        description=f"只允许受控 topic id：{', '.join(sorted(CONTROLLED_TOPICS))}"
    )
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
