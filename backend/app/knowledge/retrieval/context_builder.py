from __future__ import annotations

from app.knowledge.domain.models import (
    ConfirmedFact,
    ConversationContext,
    ConversationMessage,
)
from app.knowledge.ingestion.chunker import token_count


IMPORTANT_MARKERS = ("拒绝", "不见", "边界", "承诺", "答应", "计划", "分手", "不要")


def _fact_cost(fact: ConfirmedFact) -> int:
    return token_count(fact.content) + sum(token_count(item) for item in fact.source_message_ids)


def build_context(
    current_message: str,
    messages: list[ConversationMessage],
    conversation_summary: str | None = None,
    relationship_id: str | None = None,
    facts: list[ConfirmedFact] | None = None,
    preferences: list[ConfirmedFact] | None = None,
    token_budget: int = 2400,
) -> ConversationContext:
    facts = facts or []
    preferences = preferences or []
    candidate_facts = [
        fact for fact in facts if fact.relationship_id == relationship_id
    ][:20]
    candidate_preferences = [
        fact
        for fact in preferences
        if fact.relationship_id in {None, relationship_id}
    ][:10]
    recent = list(messages[-12:])
    for index in range(len(recent) - 1, -1, -1):
        message = recent[index]
        if message.role == "user" and message.content == current_message:
            recent.pop(index)
            break
    # Preserve chronological output while retaining explicit boundaries before small talk.
    ranked = sorted(
        enumerate(recent),
        key=lambda item: (
            not any(marker in item[1].content for marker in IMPORTANT_MARKERS),
            -item[0],
        ),
    )
    selected_indexes: set[int] = set()
    used = token_count(current_message)
    for index, message in ranked:
        cost = token_count(message.content)
        if used + cost <= token_budget:
            selected_indexes.add(index)
            used += cost
    eligible_facts: list[ConfirmedFact] = []
    for fact in candidate_facts:
        cost = _fact_cost(fact)
        if used + cost <= token_budget:
            eligible_facts.append(fact)
            used += cost
    eligible_preferences: list[ConfirmedFact] = []
    for preference in candidate_preferences:
        cost = _fact_cost(preference)
        if used + cost <= token_budget:
            eligible_preferences.append(preference)
            used += cost
    selected_summary = None
    if conversation_summary:
        cost = token_count(conversation_summary)
        if used + cost <= token_budget:
            selected_summary = conversation_summary
    selected = [
        message for index, message in enumerate(recent) if index in selected_indexes
    ]
    return ConversationContext(
        current_message=current_message,
        recent_messages=selected,
        conversation_summary=selected_summary,
        relationship_id=relationship_id,
        relationship_facts=eligible_facts,
        user_preferences=eligible_preferences,
    )
