from app.agents.assistant.state import AssistantState


def route_after_retrieval(state: AssistantState) -> str:
    if not state.get("retrieved_knowledge") and state.get("retrieval_attempt", 0) == 0:
        return "refine"
    return "generate"
