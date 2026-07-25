from __future__ import annotations

import re

import jieba


TOKENIZER_VERSION = "jieba-0.42"
STOP_WORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "你",
    "他",
    "她",
    "它",
    "和",
    "与",
    "就",
    "都",
    "而",
}
# These words carry decision direction and must survive stop-word changes.
PROTECTED_WORDS = {"不", "没", "没有", "别", "拒绝", "边界", "关系", "对方"}


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    tokens: list[str] = []
    for raw in jieba.cut(normalized, cut_all=False):
        token = raw.strip()
        if (
            not token
            or (token not in PROTECTED_WORDS and token in STOP_WORDS)
            or not re.search(r"[\w\u3400-\u9fff]", token)
        ):
            continue
        if len(token) > 1 and token[0] in {"不", "没"}:
            tokens.append(token[0])
        tokens.append(token)
    return tokens


def build_query_tokens(query: str, required_topics: list[str]) -> list[str]:
    # Repetition is intentional: PostgreSQL query construction can weight required topics.
    return tokenize(query) + [
        token
        for topic in required_topics
        for token in tokenize(topic)
        for _ in range(2)
    ]
