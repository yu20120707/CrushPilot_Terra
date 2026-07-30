from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

ReplyStyle = Literal["暧昧", "稳重", "激进"]


class ChatResult(BaseModel):
    skill: Literal["goutoujunshi"] = "goutoujunshi"
    intent: str = Field(min_length=1, max_length=80)
    judgement: str = Field(min_length=1, max_length=500)
    recommended_reply: str = Field(min_length=1, max_length=1000)
    alternatives: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        min_length=2, max_length=2
    )
    primary_style: ReplyStyle = "稳重"
    alternative_styles: list[ReplyStyle] = Field(
        default_factory=lambda: ["暧昧", "激进"], min_length=2, max_length=2
    )
    warning: str | None = Field(default=None, max_length=500)


def append_recent(
    old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return (old + new)[-12:]


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], append_recent]
    user_message: str
    device_id: str
    conversation_id: str
    relationship_id: str | None
    conversation_summary: str | None
    relationship_facts: list[dict[str, Any]]
    user_preferences: list[dict[str, Any]]
    conversation_context: dict[str, Any]
    scene_and_retrieval_plan: dict[str, Any]
    evidence_chunks: list[dict[str, Any]]
    evidence_assessment: dict[str, Any]
    retrieval_trace_id: str
    final_response: dict[str, Any]
