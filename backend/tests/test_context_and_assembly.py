import unittest

from app.knowledge.domain.models import (
    ConfirmedFact,
    ConversationMessage,
    EvidenceAssessment,
    SceneSnapshot,
)
from app.knowledge.retrieval.context_assembler import assemble_prompt
from app.knowledge.retrieval.context_builder import build_context
from app.knowledge.retrieval.expander import expand_selected


class ContextAndAssemblyTests(unittest.TestCase):
    def test_context_isolates_relationship_facts_and_keeps_boundaries(self):
        context = build_context(
            "怎么回",
            [
                ConversationMessage(role="counterpart", content="你好"),
                ConversationMessage(role="counterpart", content="不要再约我"),
            ],
            relationship_id="a",
            facts=[
                ConfirmedFact(
                    fact_id="a1",
                    relationship_id="a",
                    content="明确拒绝",
                    source_message_ids=["m1"],
                    confidence="explicit",
                ),
                ConfirmedFact(
                    fact_id="b1",
                    relationship_id="b",
                    content="其他人的事实",
                    source_message_ids=["m2"],
                    confidence="explicit",
                ),
            ],
            token_budget=16,
        )
        self.assertEqual([fact.fact_id for fact in context.relationship_facts], ["a1"])
        self.assertIn("不要再约我", [message.content for message in context.recent_messages])

    def test_expansion_only_uses_selected_neighbors_once(self):
        related = {"parent": {"chunk_id": "parent", "content": "definition"}}
        expanded = expand_selected(
            [
                {"chunk_id": "a", "parent_section_id": "parent"},
                {"chunk_id": "b", "parent_section_id": "parent"},
            ],
            related,
        )
        self.assertEqual(len(expanded[0]["expanded_context"]), 1)
        self.assertFalse(expanded[1]["expanded_context"])

    def test_context_budget_covers_all_fields_and_deduplicates_current_message(self):
        current = ConversationMessage(role="user", content="怎么回")
        fact = ConfirmedFact(
            fact_id="f1",
            relationship_id="a",
            content="明确拒绝",
            source_message_ids=["m1"],
            confidence="explicit",
        )
        context = build_context(
            "怎么回",
            [current, ConversationMessage(role="counterpart", content="普通闲聊")],
            conversation_summary="很长的摘要内容",
            relationship_id="a",
            facts=[fact],
            preferences=[fact.model_copy(update={"fact_id": "p1"})],
            token_budget=7,
        )
        self.assertNotIn(current, context.recent_messages)
        self.assertIsNone(context.conversation_summary)
        self.assertFalse(context.relationship_facts)
        self.assertFalse(context.user_preferences)

    def test_context_removes_only_the_latest_matching_current_message(self):
        repeated = ConversationMessage(role="user", content="怎么回")
        context = build_context(
            "怎么回",
            [
                repeated,
                ConversationMessage(role="counterpart", content="后来发生了新情况"),
                repeated,
            ],
            token_budget=100,
        )
        self.assertEqual(
            [message.content for message in context.recent_messages],
            ["怎么回", "后来发生了新情况"],
        )

    def test_prompt_section_order_is_fixed(self):
        scene = SceneSnapshot(
            task_type="reply",
            recommended_action="respond",
            current_event="拒绝",
            user_goal="降压",
            known_facts=[],
            uncertain_inferences=[],
            key_unknowns=[],
            counterpart_signals=[],
            explicit_boundaries=[],
            active_skill_scenarios=["explicit_rejection"],
            confidence=1,
        )
        assessment = EvidenceAssessment(
            status="insufficient",
            covered_topics=[],
            missing_topics=["boundary"],
            excluded_topic_hits=[],
            conflicting_chunk_ids=[],
            selected_chunk_ids=[],
        )
        context = build_context("怎么回", [])
        prompt = assemble_prompt([], {}, {}, scene, assessment, [], context, {})
        titles = [
            "CORE SKILL RULES",
            "ACTIVE SCENE POLICIES",
            "OUTPUT POLICY",
            "SCENE SNAPSHOT",
            "EVIDENCE ASSESSMENT",
            "SELECTED EVIDENCE CHUNKS",
            "CONVERSATION CONTEXT",
            "OUTPUT JSON SCHEMA / STREAMING CONTRACT",
        ]
        self.assertEqual([prompt.index(f"[{title}]") for title in titles], sorted(prompt.index(f"[{title}]") for title in titles))
        self.assertEqual(prompt.count('"current_message": "怎么回"'), 1)


if __name__ == "__main__":
    unittest.main()
