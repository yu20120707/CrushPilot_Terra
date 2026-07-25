import unittest

from pydantic import ValidationError

from app.knowledge.domain.models import ConversationContext, RetrievalPlan


class DomainModelTests(unittest.TestCase):
    def test_retrieval_plan_allows_only_one_to_three_queries(self):
        values = {
            "required_topics": [],
            "optional_topics": [],
            "excluded_topics": [],
            "hard_filters": {},
            "soft_preferences": {},
        }
        self.assertEqual(RetrievalPlan(queries=["a"], **values).final_top_k, 6)
        with self.assertRaises(ValidationError):
            RetrievalPlan(queries=[], **values)
        with self.assertRaises(ValidationError):
            RetrievalPlan(queries=["a", "b", "c", "d"], **values)

    def test_context_limits_recent_messages(self):
        with self.assertRaises(ValidationError):
            ConversationContext(
                current_message="now",
                recent_messages=[
                    {"role": "user", "content": str(index)} for index in range(13)
                ],
            )


if __name__ == "__main__":
    unittest.main()
