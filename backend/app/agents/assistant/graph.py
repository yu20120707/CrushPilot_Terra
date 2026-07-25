from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import AssistantNodes
from .schemas import ChatState


def build_assistant_graph(nodes: AssistantNodes, checkpointer: object) -> Any:
    workflow = StateGraph(ChatState)
    ordered = (
        ("build_context", nodes.build_context),
        ("analyze_scene_and_plan", nodes.analyze_scene_and_plan),
        ("retrieve_evidence", nodes.retrieve_evidence),
        ("generate_answer", nodes.generate_answer),
        ("validate_output", nodes.validate_output),
    )
    for name, node in ordered:
        workflow.add_node(name, node)
    workflow.add_edge(START, ordered[0][0])
    for (source, _), (target, _) in zip(ordered, ordered[1:]):
        workflow.add_edge(source, target)
    workflow.add_edge(ordered[-1][0], END)
    return workflow.compile(checkpointer=checkpointer)
