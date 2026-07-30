import unittest

from app.knowledge.retrieval.tokenizer import (
    build_query_tokens,
    build_topic_tokens,
    tokenize,
)


class TokenizerTests(unittest.TestCase):
    def test_keeps_negation_and_relationship_words(self):
        tokens = tokenize("我不是不想和对方建立关系")
        self.assertIn("不", tokens)
        self.assertIn("对方", tokens)
        self.assertIn("关系", tokens)
        self.assertNotIn("我", tokens)

    def test_query_tokens_keep_natural_negation(self):
        self.assertIn("不", build_query_tokens("请不要联系", ["explicit_rejection"]))

    def test_topic_tokens_expand_controlled_topics_without_negation_noise(self):
        tokens = build_topic_tokens(["reduce_pressure"])
        self.assertIn("降压", tokens)
        self.assertNotIn("不", tokens)


if __name__ == "__main__":
    unittest.main()
