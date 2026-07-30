from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)

ONTOLOGY = {
    "boundary": ("边界", "底线", "尊重", "越界"),
    "rejection": ("拒绝", "不愿意", "不接受", "别联系"),
    "relationship_progress": ("推进", "邀约", "表白", "升级关系"),
    "relationship_stop": ("停止", "退出", "分手", "不再联系"),
    "emotional_support": ("情绪", "安慰", "难过", "焦虑", "倾听"),
}


def enrich_metadata(
    *,
    source_path: str,
    title: str,
    heading_path: list[str],
    content: str,
) -> dict[str, Any]:
    """Create conservative offline-safe metadata from a controlled ontology."""
    text = " ".join((source_path, title, *heading_path, content)).lower()
    topics = [topic for topic, words in ONTOLOGY.items() if any(word in text for word in words)]
    stop = bool({"boundary", "rejection", "relationship_stop"} & set(topics))
    if stop:
        actions = ["stop"]
    elif "relationship_progress" in topics:
        actions = ["advance"]
    elif "emotional_support" in topics:
        actions = ["support"]
    else:
        actions = []
    knowledge_type = (
        "example"
        if any(marker in text for marker in ("案例", "example", "示例"))
        else "strategy" if topics else "source_note"
    )
    return {
        "topics": topics,
        "knowledge_type": knowledge_type,
        "action_labels": actions,
        "applicable_when": topics,
        "not_applicable_when": (
            ["用户明确拒绝或存在边界"] if actions == ["advance"] else []
        ),
    }


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    metadata = item.pop("metadata", None)
    if isinstance(metadata, dict):
        item = {**metadata, **item}
    chunk_id = item.get("chunk_id", item.get("id"))
    if not chunk_id or not item.get("document_id") or not isinstance(item.get("content"), str):
        raise ValueError("candidate is missing required chunk metadata")
    item["chunk_id"] = str(chunk_id)
    item.setdefault("title", "")
    item.setdefault("heading_path", [])
    defaults = enrich_metadata(
        source_path=str(item.get("source_path", "")),
        title=item["title"],
        heading_path=item["heading_path"],
        content=item["content"],
    )
    for key, value in defaults.items():
        if not item.get(key):
            item[key] = value
    return item


def isolate_bad_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    healthy: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            healthy.append(normalize_candidate(candidate))
        except (TypeError, ValueError) as exc:
            logger.warning("isolated invalid retrieval candidate: %s", exc)
            continue
    return healthy
