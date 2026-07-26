import unittest
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.assistant.graph import build_assistant_graph
from app.agents.assistant.nodes import AssistantNodes
from app.agents.assistant.schemas import ChatResult
from app.knowledge.domain.models import (
    RetrievalPlan,
    SceneAndRetrievalPlan,
    SceneSnapshot,
)
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.retrieval.reranker import RerankerUnavailable
from app.knowledge.retrieval.service import RetrievalService
from app.knowledge.retrieval.vector_retriever import VectorUnavailable
from app.knowledge.skill.runtime import SkillRuntime


SKILL_DIR = (
    Path(__file__).parents[1]
    / "app"
    / "agents"
    / "assistant"
    / "resources"
    / "goutoujunshi_skill"
)


def candidate(chunk_id: str, topics: list[str], action: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": f"document-{chunk_id}",
        "title": chunk_id,
        "heading_path": [],
        "content": f"{chunk_id} 的可执行原则",
        "topics": topics,
        "action_labels": [action],
        "knowledge_type": "principle",
        "token_count": 8,
        "score": 1.0,
    }


def scene_plan(
    query: str,
    topics: list[str],
    *,
    active: list[str] | None = None,
    task_type: str = "reply",
    action: str = "respond",
) -> SceneAndRetrievalPlan:
    return SceneAndRetrievalPlan(
        scene=SceneSnapshot(
            task_type=task_type,
            recommended_action=action,
            current_event=query,
            user_goal="获得尊重边界的建议",
            known_facts=[],
            uncertain_inferences=[],
            key_unknowns=[],
            counterpart_signals=[],
            explicit_boundaries=[],
            active_skill_scenarios=active or [],
            confidence=1,
        ),
        retrieval=RetrievalPlan(
            queries=[query],
            required_topics=topics,
            optional_topics=[],
            excluded_topics=[],
            hard_filters={},
            soft_preferences={},
        ),
    )


class Retriever:
    def __init__(self, values, error=None):
        self.values = values
        self.error = error

    def search(self, query, *_, **__):
        if self.error:
            raise self.error
        return self.values[query.splitlines()[0]]


class Reranker:
    model_name = "controlled-reranker"

    def __init__(self, error=None):
        self.error = error

    def score(self, _scene, _query, documents):
        if self.error:
            raise self.error
        return [1.0 - index / 100 for index in range(len(documents))]


class TraceWriter:
    def __init__(self):
        self.traces = []

    def write_trace(self, trace):
        self.traces.append(trace)


class ControlledModel:
    def __init__(self, planner, *, echo_internal_scores=False):
        self.planner = planner
        self.schemas = []
        self.echo_internal_scores = echo_internal_scores

    def __call__(self, _system, prompt, schema):
        self.schemas.append(schema)
        if schema is SceneAndRetrievalPlan:
            return self.planner(prompt)
        if self.echo_internal_scores and any(
            marker in prompt
            for marker in ('"score"', '"rrf_score"', '"rerank_score"', '"rank"')
        ):
            return ChatResult(
                intent="reply",
                judgement="看到内部评分。",
                recommended_reply="内部评分被泄漏。",
                alternatives=["不要输出。", "不要输出。"],
            )
        if '"status": "insufficient"' in prompt:
            return ChatResult(
                intent="general_advice",
                judgement="现有信息不足，无法可靠判断。",
                recommended_reply="请补充对方原话和前后文，这会改变建议。",
                alternatives=["先不要贸然行动。", "补充关系阶段后再判断。"],
            )
        return ChatResult(
            intent="reply",
            judgement="根据证据行动。",
            recommended_reply="先尊重对方的表达。",
            alternatives=["给彼此空间。", "只做一次低压力确认。"],
        )


def graph_for(planner, lexical, vector, reranker, *, echo_internal_scores=False):
    writer = TraceWriter()
    model = ControlledModel(planner, echo_internal_scores=echo_internal_scores)
    service = RetrievalService(
        lexical,
        vector,
        reranker,
        trace_writer=writer,
        metrics=RetrievalMetrics(),
    )
    nodes = AssistantNodes(
        skill_runtime=SkillRuntime(SKILL_DIR).load(),
        retrieval_service=service,
        call_json=model,
        metrics=RetrievalMetrics(),
        corpus_version="e2e-v1",
        embedding_model="controlled-embedding",
        trace_writer=writer,
    )
    writer.model = model
    return build_assistant_graph(nodes, InMemorySaver()), writer


def invoke(
    graph,
    conversation_id: str,
    messages: list[dict],
    current_message: str = "那我该怎么回？",
) -> dict:
    return graph.invoke(
        {
            "user_message": current_message,
            "conversation_id": conversation_id,
            "messages": [
                *messages,
                {"role": "user", "content": current_message},
            ],
        },
        {"configurable": {"thread_id": conversation_id}},
    )


class RetrievalE2ETests(unittest.TestCase):
    def test_recent_context_changes_the_retrieved_evidence(self):
        stop = candidate("stop-chunk", ["boundary", "rejection"], "stop")
        continue_chat = candidate(
            "continue-chunk", ["low_pressure_communication"], "respond"
        )
        values = {"stop-query": [stop], "continue-query": [continue_chat]}

        def planner(prompt):
            if "我明确拒绝继续见面" in prompt:
                return scene_plan(
                    "stop-query",
                    ["boundary", "rejection"],
                    active=["explicit_rejection"],
                    task_type="boundary",
                    action="stop",
                )
            return scene_plan(
                "continue-query",
                ["low_pressure_communication"],
            )

        graph, _ = graph_for(
            planner,
            Retriever(values),
            Retriever(values),
            Reranker(),
        )
        rejected = invoke(
            graph,
            "context-rejected",
            [{"role": "counterpart", "content": "我明确拒绝继续见面"}],
        )
        receptive = invoke(
            graph,
            "context-receptive",
            [{"role": "counterpart", "content": "我愿意继续聊聊"}],
        )

        self.assertEqual(
            [item["chunk_id"] for item in rejected["evidence_chunks"]],
            ["stop-chunk"],
        )
        self.assertEqual(
            [item["chunk_id"] for item in receptive["evidence_chunks"]],
            ["continue-chunk"],
        )

    def test_explicit_rejection_excludes_advancement_evidence(self):
        safe = candidate(
            "respect-boundary",
            ["rejection", "boundary", "reduce_pressure"],
            "stop",
        )
        unsafe = candidate("keep-pursuing", ["aggressive_pursuit"], "advance")
        values = {"rejection-query": [unsafe, safe]}
        graph, writer = graph_for(
            lambda _prompt: scene_plan(
                "rejection-query",
                ["rejection", "boundary", "reduce_pressure"],
                active=["explicit_rejection"],
                task_type="boundary",
                action="stop",
            ),
            Retriever(values),
            Retriever(values),
            Reranker(),
        )

        output = invoke(
            graph,
            "explicit-rejection",
            [{"role": "counterpart", "content": "我明确拒绝，请不要继续"}],
        )

        self.assertEqual(output["scene_and_retrieval_plan"]["scene"]["recommended_action"], "stop")
        self.assertEqual(
            [item["chunk_id"] for item in output["evidence_chunks"]],
            ["respect-boundary"],
        )
        self.assertNotIn(
            "keep-pursuing",
            writer.traces[-1]["evidence_assessment"]["selected_chunk_ids"],
        )

    def test_missing_information_returns_clarify_without_evidence(self):
        graph, writer = graph_for(
            lambda _prompt: scene_plan("clarify-query", []),
            Retriever({"clarify-query": []}),
            Retriever({"clarify-query": []}),
            Reranker(),
        )

        output = invoke(
            graph,
            "missing-information",
            [],
            "截图只有一句“随便”，没有前文。",
        )

        self.assertEqual(
            output["scene_and_retrieval_plan"]["scene"]["recommended_action"],
            "clarify",
        )
        self.assertEqual(output["evidence_assessment"]["status"], "insufficient")
        self.assertEqual(output["evidence_chunks"], [])
        self.assertIn("信息不足", output["final_response"]["judgement"])
        self.assertIn("补充", output["final_response"]["recommended_reply"])
        self.assertEqual(
            writer.traces[-1]["evidence_assessment"]["status"],
            "insufficient",
        )

    def test_reranker_failure_falls_back_through_graph_and_trace(self):
        safe = candidate("rrf-fallback", ["boundary"], "stop")
        values = {"boundary-query": [safe]}
        graph, writer = graph_for(
            lambda _prompt: scene_plan(
                "boundary-query", ["boundary"], task_type="boundary", action="stop"
            ),
            Retriever(values),
            Retriever(values),
            Reranker(RerankerUnavailable()),
        )

        output = invoke(graph, "reranker-failure", [])

        self.assertEqual(output["evidence_chunks"][0]["chunk_id"], "rrf-fallback")
        self.assertEqual(writer.traces[-1]["fallback_reason"], "reranker_unavailable")
        self.assertEqual(output["final_response"]["recommended_reply"], "先尊重对方的表达。")

    def test_embedding_failure_uses_lexical_evidence_through_graph_and_trace(self):
        safe = candidate("lexical-fallback", ["boundary"], "stop")
        graph, writer = graph_for(
            lambda _prompt: scene_plan(
                "boundary-query", ["boundary"], task_type="boundary", action="stop"
            ),
            Retriever({"boundary-query": [safe]}),
            Retriever({}, VectorUnavailable()),
            Reranker(),
        )

        output = invoke(graph, "embedding-failure", [])

        self.assertEqual(output["evidence_chunks"][0]["chunk_id"], "lexical-fallback")
        self.assertEqual(writer.traces[-1]["fallback_reason"], "vector_unavailable")
        self.assertEqual(output["final_response"]["recommended_reply"], "先尊重对方的表达。")

    def test_default_path_uses_two_model_calls_and_hides_internal_scores(self):
        safe = candidate("scored-evidence", ["boundary"], "stop")
        safe.update({"rrf_score": 0.75, "rerank_score": 0.9})
        values = {"boundary-query": [safe]}
        graph, writer = graph_for(
            lambda _prompt: scene_plan(
                "boundary-query", ["boundary"], task_type="boundary", action="stop"
            ),
            Retriever(values),
            Retriever(values),
            Reranker(),
            echo_internal_scores=True,
        )

        output = invoke(graph, "no-score-leakage", [])

        self.assertEqual(writer.model.schemas, [SceneAndRetrievalPlan, ChatResult])
        self.assertNotEqual(output["final_response"]["recommended_reply"], "内部评分被泄漏。")
        serialized_response = str(output["final_response"])
        self.assertNotIn("score", serialized_response)
        self.assertNotIn("0.75", serialized_response)
        self.assertNotIn("0.9", serialized_response)


if __name__ == "__main__":
    unittest.main()
