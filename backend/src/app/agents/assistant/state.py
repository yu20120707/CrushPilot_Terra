from typing import Annotated, Any, TypedDict


def keep_recent_messages(
    existing: list[dict[str, str]],
    incoming: list[dict[str, str]],
) -> list[dict[str, str]]:
    return (existing + incoming)[-12:]


class AssistantState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], keep_recent_messages]
    user_message: str
    device_id: str
    intent_analysis: dict[str, Any]
    retrieval_queries: list[str]
    retrieval_attempt: int
    retrieved_knowledge: list[dict[str, str]]
    final_response: dict[str, Any]
