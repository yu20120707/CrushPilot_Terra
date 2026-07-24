from pathlib import Path

from app.agents.assistant.tools import (
    RAW_KNOWLEDGE_DIR,
    SAFE_CARD_IDS,
    SAFE_CARD_INDEX,
    SOURCE_ROOT,
    UPSTREAM_COMMIT,
    build_research_cards,
    build_research_vector_index,
    read_cards,
    reference_context,
    research_manifest,
    retrieve_by_keywords,
    retrieve_by_vector,
    selected_reference_paths,
)


def test_safe_card_retrieval_covers_income_and_fatigue() -> None:
    assert retrieve_by_keywords(["工资", "经济条件"])[0]["source"] == "card:income-demeaning"
    assert retrieve_by_keywords(["累"])[0]["source"] == "card:low-pressure-chat"


def test_alias_retrieval_handles_fatigue_without_exact_word() -> None:
    index = [
        {
            "source": "reply-suggestion.md",
            "title": "回复",
            "content": "听起来今天挺累的，回去早点休息。",
            "tier": "safe",
        }
    ]
    assert retrieve_by_keywords(["疲惫"], index) == index


def test_vector_retrieval_is_off_until_local_index_exists(tmp_path: Path) -> None:
    assert retrieve_by_vector("她嫌我工资低", index_path=tmp_path / "missing.json") == []


def test_safe_index_contains_only_approved_cards() -> None:
    assert {card["id"] for card in read_cards()} == SAFE_CARD_IDS
    assert {chunk["source"] for chunk in SAFE_CARD_INDEX} == {
        f"card:{card_id}" for card_id in SAFE_CARD_IDS
    }


def test_every_raw_document_has_research_card_and_embedding(tmp_path: Path) -> None:
    cards_path = tmp_path / "cards.json"
    cards = build_research_cards(cards_path)
    index = build_research_vector_index(
        tmp_path / "vectors.json",
        encode_fn=lambda texts: [[float(index)] for index, _ in enumerate(texts)],
        cards_path=cards_path,
    )
    assert len(cards) == len(list(RAW_KNOWLEDGE_DIR.rglob("*.md")))
    assert len(research_manifest()) == len(cards)
    assert len(index["cards"]) == len(cards)
    assert index["scope"] == "research_only"


def test_vendored_skill_and_routed_references_are_preserved() -> None:
    assert UPSTREAM_COMMIT == "81d99da388da11cb6c1bc18259c6802c82fbaf41"
    assert (SOURCE_ROOT / "LICENSE").is_file()
    assert "PolyForm Noncommercial" in (SOURCE_ROOT / "LICENSE").read_text(
        encoding="utf-8"
    )
    paths = selected_reference_paths("她说你这点工资够干啥呀", limit=3)
    assert [path.name for path in paths] == [
        "12-金钱家务育儿与双方家庭.md",
        "06-吸引约会与关系启动.md",
    ]
    references = reference_context("她说你这点工资够干啥呀")
    assert "12-金钱家务育儿与双方家庭.md" in references
    assert "01-证据分级与内容边界.md" not in references
