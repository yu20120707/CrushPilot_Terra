import unittest

from app.knowledge.retrieval.tokenizer import build_query_tokens, tokenize


class TokenizerTests(unittest.TestCase):
    def test_keeps_negation_and_relationship_words(self):
        tokens = tokenize("我不是不想和对方建立关系")
        self.assertIn("不", tokens)
        self.assertIn("对方", tokens)
        self.assertIn("关系", tokens)
        self.assertNotIn("我", tokens)

    def test_required_topics_receive_extra_weight(self):
        tokens = build_query_tokens("如何回复", ["明确拒绝"])
        self.assertGreaterEqual(tokens.count("拒绝"), 2)


if __name__ == "__main__":
    unittest.main()
