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
        self.assertIn("required_topics 是可组合的强制检索覆盖集合", prompt)
        self.assertIn("非 insufficient_information 场景中", prompt)
        self.assertIn("不得用 uncertainty 或 conservative_action 替代可检索 topic", prompt)
        self.assertIn("判断主动、投入或长期回应时包含 reciprocity", prompt)
        self.assertIn("延迟、已读或线上消息时包含 digital_context", prompt)
        self.assertIn("表达需要空间、不舒服、停止或隐私时包含 boundary", prompt)
        self.assertIn("结束、分手或前任断联时包含 relationship_exit", prompt)
        self.assertIn(
            "insufficient_information 仅在无法给出任何有证据支持的低风险下一步时选择",
            prompt,
        )
        self.assertIn("task_type 按用户主要请求判断", prompt)
        self.assertIn("recommended_action 按下一步行动分别判断", prompt)
        self.assertIn("primary query 会作为唯一 Reranker query", prompt)
        self.assertIn("必须独立包含关键主体、行为、否定、关系状态", prompt)
        self.assertIn("不得改成泛化标题", prompt)
        self.assertIn("secondary query 只补充不同检索角度", prompt)

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
            (
                "她只回了一个嗯，除此之外没有上下文。",
                [],
                ("general_advice", "clarify", ["insufficient_information"]),
            ),
            (
                "朋友转述说他可能有好感，但我没看到原话。",
                [],
                ("general_advice", "clarify", ["insufficient_information"]),
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

        model_plan = plan()
        model_plan.scene.task_type = "relationship_analysis"
        model_plan.scene.recommended_action = "observe"
        model_plan.scene.active_skill_scenarios = ["insufficient_information"]
        model_plan.retrieval.required_topics = ["mind_reading"]
        model_plan.retrieval.excluded_topics = ["mind_reading"]
        nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=Mock(return_value=model_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        state = {
            "user_message": "朋友转述说他可能对我有好感，但我没看到原话。",
            "conversation_id": "normalize-insufficient",
            "messages": [{"role": "user", "content": "朋友转述说他可能对我有好感，但我没看到原话。"}],
        }
        state.update(nodes.build_context(state))
        output = nodes.analyze_scene_and_plan(state)["scene_and_retrieval_plan"]
        self.assertEqual(output["scene"]["task_type"], "general_advice")
        self.assertEqual(output["scene"]["recommended_action"], "clarify")
        self.assertEqual(
            output["scene"]["active_skill_scenarios"],
            ["insufficient_information"],
        )
        self.assertEqual(
            output["retrieval"]["required_topics"],
            ["uncertainty", "conservative_action"],
        )
        self.assertEqual(output["retrieval"]["excluded_topics"], ["mind_reading"])
        self.assertFalse(
            set(output["retrieval"]["required_topics"])
            & set(output["retrieval"]["excluded_topics"])
        )

        rejection_plan = plan()
        rejection_plan.scene.active_skill_scenarios = ["explicit_rejection"]
        rejection_plan.retrieval.required_topics = ["boundary"]
        rejection_plan.retrieval.excluded_topics = ["boundary", "reduce_pressure"]
        rejection_nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=Mock(return_value=rejection_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        rejection_state = {
            "user_message": "对方明确拒绝我了。",
            "conversation_id": "policy-required-wins",
            "messages": [{"role": "user", "content": "对方明确拒绝我了。"}],
        }
        rejection_state.update(rejection_nodes.build_context(rejection_state))
        rejection_output = rejection_nodes.analyze_scene_and_plan(rejection_state)[
            "scene_and_retrieval_plan"
        ]["retrieval"]
        self.assertTrue(
            {"rejection", "boundary", "reduce_pressure"}.issubset(
                rejection_output["required_topics"]
            )
        )
        self.assertFalse(
            {"rejection", "boundary", "reduce_pressure"}
            & set(rejection_output["excluded_topics"])
        )
        self.assertFalse(
            set(rejection_output["required_topics"])
            & set(rejection_output["excluded_topics"])
        )

        support_plan = plan()
        support_plan.scene.task_type = "emotional_support"
        support_plan.scene.recommended_action = "support"
        support_plan.scene.active_skill_scenarios = ["emotional_support"]
        support_nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=Mock(return_value=support_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        support_state = {
            "user_message": "我没看到她之前发的原话，但她现在哭着说自己很失败，我该怎么陪她？",
            "conversation_id": "active-scene-beats-missing-fallback",
            "messages": [
                {
                    "role": "user",
                    "content": "我没看到她之前发的原话，但她现在哭着说自己很失败，我该怎么陪她？",
                }
            ],
        }
        support_state.update(support_nodes.build_context(support_state))
        support_output = support_nodes.analyze_scene_and_plan(support_state)[
            "scene_and_retrieval_plan"
        ]
        self.assertEqual(support_output["scene"]["task_type"], "emotional_support")
        self.assertEqual(support_output["scene"]["recommended_action"], "support")
        self.assertEqual(
            support_output["scene"]["active_skill_scenarios"],
            ["emotional_support"],
        )
        self.assertIn(
            "emotional_support",
            support_output["retrieval"]["required_topics"],
        )

        exit_plan = plan()
        exit_plan.scene.task_type = "relationship_exit"
        exit_plan.scene.recommended_action = "exit"
        exit_plan.scene.active_skill_scenarios = ["explicit_rejection"]
        exit_nodes = AssistantNodes(
            skill_runtime=runtime,
            retrieval_service=FakeRetrieval(),
            call_json=Mock(return_value=exit_plan),
            metrics=RetrievalMetrics(),
            corpus_version="test-v1",
            embedding_model="embed-test",
        )
        exit_state = {
            "user_message": "我已决定结束关系，想处理后续退出。",
            "conversation_id": "preserve-exit",
            "messages": [{"role": "user", "content": "我已决定结束关系，想处理后续退出。"}],
        }
        exit_state.update(exit_nodes.build_context(exit_state))
        exit_scene = exit_nodes.analyze_scene_and_plan(exit_state)[
            "scene_and_retrieval_plan"
        ]["scene"]
        self.assertEqual(exit_scene["task_type"], "relationship_exit")
        self.assertEqual(exit_scene["recommended_action"], "exit")

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
        self.assertIn("命中时 action 选 clarify", scene_prompt)
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
        for index, message in enumerate(("怎么跟踪她", "怎么追求未成年女孩"), 1):
            with self.subTest(message=message):
                output = graph.invoke(
                    {
                        "user_message": message,
                        "conversation_id": f"conversation-risk-{index}",
                        "messages": [{"role": "user", "content": message}],
                    },
                    {"configurable": {"thread_id": f"thread-risk-{index}"}},
                )
                self.assertEqual(
                    output["final_response"]["skill"], "goutoujunshi"
                )
                self.assertIsNotNone(output["final_response"]["warning"])
        self.assertEqual(model.call_count, 0)

    def test_stopping_harmful_behavior_uses_normal_planner(self):
        messages = (
            "前任要求不再联系，我需要停止纠缠。",
            "我需要停止对前任的纠缠。",
            "请不要再去跟踪她。",
            "必须停止继续对她的骚扰。",
            "我决定不再通过小号骚扰她。",
            "我想摆脱纠缠对方的冲动。",
        )
        for index, message in enumerate(messages, 1):
            with self.subTest(message=message):
                model_plan = plan()
                model = Mock(
                    side_effect=[
                        model_plan,
                        ChatResult(
                            intent="boundary",
                            judgement="应当尊重对方边界。",
                            recommended_reply="我会停止纠缠，也不会再联系你。",
                            alternatives=["尊重你的决定。", "我不会继续打扰。"],
                        ),
                    ]
                )
                graph = self.make_graph(model)
                output = graph.invoke(
                    {
                        "user_message": message,
                        "conversation_id": f"conversation-stop-harm-{index}",
                        "messages": [{"role": "user", "content": message}],
                    },
                    {
                        "configurable": {
                            "thread_id": f"thread-stop-harm-{index}"
                        }
                    },
                )

                self.assertEqual(model.call_count, 2)
                self.assertEqual(
                    output["scene_and_retrieval_plan"]["retrieval"]["queries"],
                    model_plan.retrieval.queries,
                )

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
