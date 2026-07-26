from __future__ import annotations

from typing import Literal

from app.knowledge.domain.models import (
    CONTROLLED_TOPICS,
    PLANNING_ONLY_TOPICS,
    TOPIC_SEARCH_TERMS,
)
from pydantic import BaseModel, ConfigDict, field_validator


ACTION_LABELS = {
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
}


class SuggestedMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    knowledge_type: Literal[
        "hard_rule",
        "principle",
        "strategy",
        "example",
        "counterexample",
        "source_note",
    ]
    topics: list[str]
    action_labels: list[str]
    applicable_when: list[str]
    not_applicable_when: list[str]

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, values: list[str]) -> list[str]:
        if any(value not in CONTROLLED_TOPICS for value in values):
            raise ValueError("topics must use controlled topic ids")
        return cls.validate_string_list(values)

    @field_validator("action_labels")
    @classmethod
    def validate_actions(cls, values: list[str]) -> list[str]:
        if any(value not in ACTION_LABELS for value in values):
            raise ValueError("action_labels must use controlled action ids")
        return cls.validate_string_list(values)

    @field_validator("applicable_when", "not_applicable_when")
    @classmethod
    def validate_string_list(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("metadata lists cannot contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("metadata lists cannot contain duplicates")
        return values


TOPIC_RULES = TOPIC_SEARCH_TERMS


def suggest_metadata(title: str, heading_path: list[str], content: str) -> SuggestedMetadata:
    text = "\n".join((title, *heading_path, content))
    lowered = text.lower()
    structural_text = "\n".join((title, *heading_path)).lower()
    topics = [
        topic
        for topic, keywords in TOPIC_RULES.items()
        if any(
            keyword.lower()
            in (structural_text if topic == "manipulation" else lowered)
            for keyword in keywords
        )
    ] or ["general_relationship_advice"]
    if any(keyword in text for keyword in ("禁止", "不得", "风险", "红线", "危机")):
        knowledge_type = "hard_rule"
    elif any(keyword in text for keyword in ("案例", "示例", "对话", "话术")):
        knowledge_type = "example"
    elif any(keyword in text for keyword in ("步骤", "方法", "策略", "实践", "指南")):
        knowledge_type = "strategy"
    else:
        knowledge_type = "principle"

    actions: list[str] = []
    not_applicable: list[str] = []
    if set(topics) & {"rejection", "boundary", "reduce_pressure", "relationship_exit"}:
        actions.extend(("reduce_pressure", "stop"))
    if "invitation" in topics and not set(topics) & {"rejection", "boundary"}:
        actions.append("advance")
        not_applicable.extend(("explicit_rejection", "explicit_boundary"))
    if "manipulation" in topics:
        actions.append("stop")
        not_applicable.append("normal_online_advice")
    if "emotional_support" in topics:
        actions.append("support")
    return SuggestedMetadata(
        knowledge_type=knowledge_type,
        topics=list(dict.fromkeys(topics)),
        action_labels=list(dict.fromkeys(actions)),
        applicable_when=list(dict.fromkeys(topics)),
        not_applicable_when=list(dict.fromkeys(not_applicable)),
    )
