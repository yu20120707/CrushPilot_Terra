import unittest

from app.knowledge.ingestion.metadata_enricher import (
    CONTROLLED_TOPICS,
    suggest_metadata,
)


class MetadataEnricherTests(unittest.TestCase):
    def test_safety_warning_does_not_misclassify_boundary_content_as_manipulation(self):
        metadata = suggest_metadata(
            "同意、边界、性与亲密",
            ["同意是持续过程"],
            "同意必须自由，没有威胁、操控或强迫。",
        )
        self.assertIn("boundary", metadata.topics)
        self.assertNotIn("manipulation", metadata.topics)

    def test_manipulation_heading_remains_excludable(self):
        metadata = suggest_metadata(
            "风险材料",
            ["PUA 常见操控技术风险表"],
            "这些做法不可采用。",
        )
        self.assertIn("manipulation", metadata.topics)
    def test_rejection_boundary_produces_stop_direction(self):
        metadata = suggest_metadata(
            "明确拒绝", ["沟通", "边界"], "对方拒绝后应降低压力，不再推进。"
        )
        self.assertIn("rejection", metadata.topics)
        self.assertIn("boundary", metadata.topics)
        self.assertIn("stop", metadata.action_labels)
        self.assertNotIn("advance", metadata.action_labels)

    def test_invitation_is_not_applicable_after_rejection(self):
        metadata = suggest_metadata("邀约指南", ["约会"], "如何提出见面邀请")
        self.assertIn("advance", metadata.action_labels)
        self.assertIn("explicit_rejection", metadata.not_applicable_when)

    def test_manipulation_is_marked_out_of_normal_online_advice(self):
        metadata = suggest_metadata("PUA操控", ["风险"], "禁止欺骗和强迫")
        self.assertIn("manipulation", metadata.topics)
        self.assertIn("normal_online_advice", metadata.not_applicable_when)
        self.assertEqual(metadata.knowledge_type, "hard_rule")

    def test_digital_delay_and_short_reply_use_controlled_topics(self):
        delayed = suggest_metadata(
            "文字沟通的局限", ["在线关系"], "延迟回复和晚回不能单独证明态度。"
        )
        short = suggest_metadata(
            "对方只回哈哈", ["回复"], "不追问，先收尾。"
        )
        self.assertIn("digital_context", delayed.topics)
        self.assertIn("reduce_pressure", short.topics)

    def test_planning_and_corpus_topics_share_one_controlled_ontology(self):
        self.assertTrue(
            {
                "aggressive_pursuit",
                "conservative_action",
                "uncertainty",
            }
            <= CONTROLLED_TOPICS
        )


if __name__ == "__main__":
    unittest.main()
