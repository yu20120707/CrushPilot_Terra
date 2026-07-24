from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.assistant.edges import route_after_retrieval
from app.agents.assistant.nodes import AssistantNodes, ModelClient
from app.agents.assistant.state import AssistantState
from app.core.config import Settings, get_settings
from app.infrastructure.model_factory import create_model_client


def build_graph(
    *,
    settings: Settings | None = None,
    model_client: ModelClient | None = None,
    checkpointer: Any | None = None,
) -> Any:
    settings = settings or get_settings()
    model_client = model_client or create_model_client(settings)
    nodes = AssistantNodes(settings, model_client)

    workflow = StateGraph(AssistantState)
    workflow.add_node("analyze_intent", nodes.analyze_intent)
    workflow.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    workflow.add_node("refine_query", nodes.refine_query)
    workflow.add_node("generate_reply", nodes.generate_reply)

    workflow.add_edge(START, "analyze_intent")
    workflow.add_edge("analyze_intent", "retrieve_knowledge")
    workflow.add_conditional_edges(
        "retrieve_knowledge",
        route_after_retrieval,
        {"refine": "refine_query", "generate": "generate_reply"},
    )
    workflow.add_edge("refine_query", "retrieve_knowledge")
    workflow.add_edge("generate_reply", END)

    if checkpointer is None:
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer)


# LangGraph CLI/Studio entry point. FastAPI builds its own persisted instance
# during application startup.
graph = build_graph()
