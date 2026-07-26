import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ["DEMO_MODE"] = "true"

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.assistant.graph import build_assistant_graph
from app.agents.assistant.nodes import AssistantNodes, MISSING_CONTEXT_PATTERN
from app.agents.assistant.prompts import scene_plan_prompt
from app.agents.assistant.schemas import ChatResult
from app.knowledge.domain.models import (
    TOPIC_PLANNING_DESCRIPTIONS,
    TOPIC_SEARCH_TERMS,
    ConversationContext,
    EvidenceAssessment,
    RetrievalPlan,
    RetrievalTrace,
    SceneAndRetrievalPlan,
    SceneSnapshot,
)
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.retrieval.service import RetrievalResult
from app.knowledge.skill.runtime import SkillRuntime


SKILL_DIR = (
    Path(__file__).parents[1]
    / "app"
    / "agents"
    / "assistant"
    / "resources"
    / "goutoujunshi_skill"
)


def plan() -> SceneAndRetrievalPlan:
    return SceneAndRetrievalPlan(
        scene=SceneSnapshot(
            task_type="reply",
            recommended_action="respond",
            current_event="对方说今天很累",
            user_goal="低压力回应",
            known_facts=["对方说今天很累"],
            uncertain_inferences=[],
            key_unknowns=[],
            counterpart_signals=["疲惫"],
            explicit_boundaries=[],
            active_skill_scenarios=[],
            confidence=0.9,
        ),
        retrieval=RetrievalPlan(
            queries=["对方疲惫时如何低压力回应"],
            required_topics=[],
            optional_topics=["emotional_support"],
            excluded_topics=["manipulation"],
            hard_filters={"review_status": ["approved"]},
            soft_preferences={"task_types": ["reply"]},
        ),
    )


class FakeRetrieval:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        chunks = [
            {
                "chunk_id": "selected-1",
                "title": "低压力回应",
                "heading_path": ["低压力回应"],
                "content": "先接住情绪，允许对方晚些回复。",
                "knowledge_type": "strategy",
                "topics": ["emotional_support"],
                "token_count": 14,
                "rerank_score": 0.9,
            }
        ]
        assessment = EvidenceAssessment(
            status="sufficient",
            covered_topics=[],
            missing_topics=[],
            excluded_topic_hits=[],
            conflicting_chunk_ids=[],
            selected_chunk_ids=["selected-1"],
        )
        trace = RetrievalTrace(
            request_id=kwargs["request_id"],
            conversation_id=kwargs["conversation_id"],
            skill_version=kwargs["skill_version"],
            skill_sha256=kwargs["skill_sha256"],
            corpus_version=kwargs["corpus_version"],
            scene_snapshot={},
            retrieval_plan={},
            lexical_candidates=[],
            vector_candidates=[],
            fused_candidates=[],
            reranked_candidates=[],
            selected_chunks=[],
            evidence_assessment=assessment.model_dump(),
            latency_ms={},
        )
        return RetrievalResult(chunks, assessment, trace)


class AgentIntegrationTests(unittest.TestCase):
    def test_scene_prompt_explains_controlled_topic_ids(self):
        runtime = SkillRuntime(SKILL_DIR).load()
        prompt = scene_plan_prompt(
            ConversationContext(current_message="对方第二天才回", recent_messages=[]),
            runtime.view([]),
        )

        glossary = json.loads(
            prompt.split("[TOPIC GLOSSARY]\n", 1)[1].split(
                "\n\n[CONVERSATION CONTEXT]", 1
            )[0]
        )
        self.assertEqual(set(glossary), set(TOPIC_SEARCH_TERMS))
        self.assertEqual(
            glossary["digital_context"],
            {
                "meaning": TOPIC_PLANNING_DESCRIPTIONS["digital_context"],
                "search_terms": list(TOPIC_SEARCH_TERMS["digital_context"]),
            },
        )

    def make_graph(self, call_json, trace_writer=None):
        runtime = SkillRuntime(SKILL_DIR).load()
        nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=call_json,
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
            trace_writer=trace_writer,
        )
        return build_assistant_graph(nodes, InMemorySaver())

    def test_model_hard_filters_are_sanitized_before_retrieval(self):
        runtime = SkillRuntime(SKILL_DIR).load()
        retrieval = FakeRetrieval()
        model_plan = plan()
        model_plan.retrieval.hard_filters = {
            "relationship_type": ["dating"],
            "task_types": ["reply"],
            "review_status": ["draft"],
            "knowledge_type": [],
        }
        model_plan.retrieval.required_topics = ["emotional_support", "free text"]
        model_plan.retrieval.excluded_topics = ["manipulation", "未知标签"]
        nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=retrieval,
            call_json=Mock(return_value=model_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        state = {
            "user_message": "她说今天很累，怎么回？",
            "conversation_id": "conversation-filter",
            "messages": [{"role": "user", "content": "她说今天很累，怎么回？"}],
        }
        state.update(nodes.build_context(state))
        state.update(nodes.analyze_scene_and_plan(state))
        nodes.retrieve_evidence(state)

        hard_filters = retrieval.calls[0]["plan"].hard_filters
        self.assertEqual(
            hard_filters,
            {
                "task_types": ["reply"],
                "review_status": ["approved"],
            },
        )
        self.assertEqual(
            retrieval.calls[0]["plan"].required_topics,
            ["emotional_support"],
        )
        self.assertEqual(
            retrieval.calls[0]["plan"].excluded_topics,
            ["manipulation"],
        )

    def test_known_facts_require_input_provenance(self):
        runtime = SkillRuntime(SKILL_DIR).load()
        model_plan = plan()
        model_plan.scene.known_facts = [
            "对方说今天很累",
            "对方说今天很累并答应周末见面",
            "对方已经答应周末见面",
        ]
        nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=Mock(return_value=model_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        state = {
            "user_message": "我该怎么回？",
            "conversation_id": "known-fact-provenance",
            "messages": [
                {"role": "counterpart", "content": "对方说：今天很累。"},
                {"role": "user", "content": "我该怎么回？"},
            ],
        }
        state.update(nodes.build_context(state))

        scene = nodes.analyze_scene_and_plan(state)["scene_and_retrieval_plan"]["scene"]

        self.assertEqual(scene["known_facts"], ["对方说今天很累"])
        self.assertIn(
            "对方说今天很累并答应周末见面",
            scene["key_unknowns"],
        )
        self.assertIn("对方已经答应周末见面", scene["key_unknowns"])

    def test_missing_context_and_explicit_rejection_are_normalized(self):
        runtime = SkillRuntime(SKILL_DIR).load()
        for message, active, expected in (
            (
                "只看到一个表情，缺少双方关系阶段。",
                [],
                ("general_advice", "clarify", ["insufficient_information"]),
            ),
            (
                "对方明确拒绝我了，我不知道怎么回复。",
                ["explicit_rejection"],
                ("boundary", "stop", ["explicit_rejection"]),
            ),
        ):
            retrieval = FakeRetrieval()
            model_plan = plan()
            model_plan.scene.active_skill_scenarios = active
            nodes = AssistantNodes(
                skill_runtime=runtime,
                retrieval_service=retrieval,
                call_json=Mock(return_value=model_plan),
                metrics=RetrievalMetrics(),
                corpus_version="test-v1",
                embedding_model="embed-test",
            )
            state = {
                "user_message": message,
                "conversation_id": "normalize-scene",
                "messages": [{"role": "user", "content": message}],
            }
            state.update(nodes.build_context(state))
            output = nodes.analyze_scene_and_plan(state)[
                "scene_and_retrieval_plan"
            ]
            self.assertEqual(
                (
                    output["scene"]["task_type"],
                    output["scene"]["recommended_action"],
                    output["scene"]["active_skill_scenarios"],
                ),
                expected,
            )

        for message in (
            "不知道怎么安慰她，她说考试失败了。",
            "对方只说最近忙，下次吧。",
            "对方只问我周末有空吗。",
            "我们没有冲突，关系很好，我想约她周末喝咖啡。",
            "她发了晚安，不知道对方消息该怎么回。",
            "我不知道多久才能走出分手，但经过和边界都说清楚了。",
        ):
            self.assertIsNone(MISSING_CONTEXT_PATTERN.search(message))

    def test_normal_path_uses_exactly_two_llm_calls_and_selected_chunks_only(self):
        result = ChatResult(
            intent="ignored",
            judgement="先共情。",
            recommended_reply="辛苦了，先休息，忙完再聊。",
            alternatives=["不用急着回复。", "照顾好自己。"],
        )
        model = Mock(side_effect=[plan(), result])
        graph = self.make_graph(model)
        output = graph.invoke(
            {
                "user_message": "她说今天很累，怎么回？",
                "conversation_id": "conversation-1",
                "messages": [{"role": "user", "content": "她说今天很累，怎么回？"}],
            },
            {"configurable": {"thread_id": "thread-1"}},
        )
        self.assertEqual(model.call_count, 2)
        scene_prompt = model.call_args_list[0].args[1]
        self.assertIn('"scene"', scene_prompt)
        self.assertIn('"retrieval"', scene_prompt)
        self.assertIn("顶层字段必须严格为 scene 和 retrieval", scene_prompt)
        self.assertIn("primary query 优先贴近用户原话", scene_prompt)
        self.assertIn("只有出现明确拒绝时才选 explicit_rejection", scene_prompt)
        self.assertIn("此时 action 选 clarify", scene_prompt)
        final_prompt = model.call_args_list[1].args[1]
        self.assertIn("selected-1", final_prompt)
        self.assertNotIn("SKILL.md", final_prompt)
        self.assertNotIn("goutoujunshi_source", final_prompt)
        self.assertEqual(output["final_response"]["skill"], "goutoujunshi")

    def test_context_is_recent_and_relationship_isolated(self):
        model = Mock(
            side_effect=[
                plan(),
                ChatResult(
                    intent="reply",
                    judgement="判断",
                    recommended_reply="回复",
                    alternatives=["备选一", "备选二"],
                ),
            ]
        )
        graph = self.make_graph(model)
        messages = [
            {"role": "counterpart", "content": f"消息{i}"}
            for i in range(15)
        ]
        output = graph.invoke(
            {
                "user_message": "怎么回",
                "conversation_id": "conversation-2",
                "relationship_id": "current",
                "relationship_facts": [
                    {
                        "fact_id": "keep",
                        "relationship_id": "current",
                        "content": "当前关系事实",
                        "source_message_ids": ["m1"],
                        "confidence": "explicit",
                    },
                    {
                        "fact_id": "drop",
                        "relationship_id": "other",
                        "content": "其他关系事实",
                        "source_message_ids": ["m2"],
                        "confidence": "explicit",
                    },
                ],
                "messages": messages,
            },
            {"configurable": {"thread_id": "thread-2"}},
        )
        context = output["conversation_context"]
        self.assertLessEqual(len(context["recent_messages"]), 12)
        self.assertEqual(
            [fact["fact_id"] for fact in context["relationship_facts"]], ["keep"]
        )

    def test_high_risk_shortcut_keeps_skill_and_skips_llm(self):
        model = Mock()
        graph = self.make_graph(model)
        output = graph.invoke(
            {
                "user_message": "怎么跟踪她",
                "conversation_id": "conversation-3",
                "messages": [{"role": "user", "content": "怎么跟踪她"}],
            },
            {"configurable": {"thread_id": "thread-3"}},
        )
        self.assertEqual(model.call_count, 0)
        self.assertEqual(output["final_response"]["skill"], "goutoujunshi")
        self.assertIsNotNone(output["final_response"]["warning"])

    def test_scene_failure_writes_controlled_trace(self):
        writer = Mock()
        graph = self.make_graph(
            Mock(side_effect=RuntimeError("structured output failed")), writer
        )
        with self.assertRaises(RuntimeError):
            graph.invoke(
                {
                    "user_message": "怎么回",
                    "conversation_id": "conversation-failure",
                    "messages": [{"role": "user", "content": "怎么回"}],
                },
                {"configurable": {"thread_id": "thread-failure"}},
            )
        trace = writer.write_trace.call_args.args[0]
        self.assertEqual(trace["fallback_reason"], "scene_analysis_failed")
        self.assertNotIn("怎么回", str(trace))

    def test_retrieval_failure_uses_generic_stage_latency(self):
        writer = Mock()
        runtime = SkillRuntime(SKILL_DIR).load()
        retrieval = Mock()
        retrieval.retrieve.side_effect = RuntimeError("failure after lexical")
        metrics = RetrievalMetrics()
        nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=retrieval,
            call_json=Mock(return_value=plan()),
            metrics=metrics,
            corpus_version="test-v1",
            embedding_model="embed-test",
            trace_writer=writer,
        )
        graph = build_assistant_graph(nodes, InMemorySaver())
        with self.assertRaises(RuntimeError):
            graph.invoke(
                {
                    "user_message": "怎么回",
                    "conversation_id": "conversation-retrieval-failure",
                    "messages": [{"role": "user", "content": "怎么回"}],
                },
                {"configurable": {"thread_id": "thread-retrieval-failure"}},
            )
        trace = writer.write_trace.call_args.args[0]
        self.assertEqual(trace["fallback_reason"], "retrieval_failed")
        self.assertGreater(trace["latency_ms"]["retrieval_failure_latency_ms"], 0)
        self.assertNotIn("lexical_retrieval_latency_ms", trace["latency_ms"])
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["lexical_retrieval_latency_ms"]["count"], 0)
        self.assertEqual(snapshot["retrieval_failure_latency_ms"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
