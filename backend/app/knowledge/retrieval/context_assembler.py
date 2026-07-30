from __future__ import annotations

import json

from app.knowledge.domain.models import (
    ConversationContext,
    EvidenceAssessment,
    SceneSnapshot,
)
from app.knowledge.ingestion.chunker import token_count


INTERNAL_EVIDENCE_FIELDS = {
    "score",
    "rrf_score",
    "rerank_score",
    "rank",
    "lexical_rank",
    "vector_rank",
}


def _without_internal_scores(value):
    if isinstance(value, dict):
        return {
            key: _without_internal_scores(item)
            for key, item in value.items()
            if key not in INTERNAL_EVIDENCE_FIELDS
        }
    if isinstance(value, list):
        return [_without_internal_scores(item) for item in value]
    return value


def _evidence_within_budget(evidence: list[dict], budget: int) -> list[dict]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.get("knowledge_type") == "example",
            -float(item.get("rerank_score", item.get("rrf_score", 0))),
        ),
    )
    selected: list[dict] = []
    used = 0
    for item in ordered:
        cost = token_count(str(item.get("content", ""))) + sum(
            token_count(str(context.get("content", "")))
            for context in item.get("expanded_context", [])
        )
        if used + cost <= budget:
            selected.append(_without_internal_scores(item))
            used += cost
    return selected


def assemble_prompt(
    core_skill_rules: list[dict],
    active_scene_policies: dict[str, dict],
    output_policy: dict,
    scene: SceneSnapshot,
    assessment: EvidenceAssessment,
    evidence: list[dict],
    context: ConversationContext,
    output_schema: dict,
    evidence_budget: int = 3500,
) -> str:
    sections = [
        ("CORE SKILL RULES", core_skill_rules),
        ("ACTIVE SCENE POLICIES", active_scene_policies),
        ("OUTPUT POLICY", output_policy),
        ("SCENE SNAPSHOT", scene.model_dump()),
        ("EVIDENCE ASSESSMENT", assessment.model_dump()),
        ("SELECTED EVIDENCE CHUNKS", _evidence_within_budget(evidence, evidence_budget)),
        (
            "CONVERSATION CONTEXT",
            context.model_dump(mode="json", exclude={"current_message"}),
        ),
        ("USER CURRENT MESSAGE", context.current_message),
        ("OUTPUT JSON SCHEMA / STREAMING CONTRACT", output_schema),
    ]
    return "\n\n".join(
        f"[{title}]\n{json.dumps(value, ensure_ascii=False)}"
        for title, value in sections
    )
