"""Safe knowledge retrieval and reference tools used by the assistant graph.

Raw material under ``knowledge/pua-knowledge-sharing`` is retained for research
provenance only. It is never embedded or returned to the chat model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agents.assistant.prompts import SYSTEM_PROMPT
from app.core.config import get_settings
from app.infrastructure.model_factory import create_model_client

SETTINGS = get_settings()
KNOWLEDGE_DIR = SETTINGS.knowledge_dir
RAW_KNOWLEDGE_DIR = KNOWLEDGE_DIR / "pua-knowledge-sharing"
CARDS_PATH = KNOWLEDGE_DIR / "cards" / "approved.json"
DATA_DIR = SETTINGS.data_dir
VECTOR_INDEX_PATH = DATA_DIR / "knowledge_vectors.json"
GENERATED_CARDS_PATH = DATA_DIR / "generated_cards.json"
RESEARCH_CARDS_PATH = DATA_DIR / "research_cards.json"
RESEARCH_VECTOR_INDEX_PATH = DATA_DIR / "research_vectors.json"
EMBEDDING_MODEL = SETTINGS.embedding_model
CARD_UNSAFE_PATTERN = ("跟踪", "尾随", "蹲守", "威胁", "恐吓", "强迫", "骚扰", "灌醉", "下药", "偷拍", "冒充")
TERM_ALIASES = {
    "疲惫": ["累", "辛苦", "休息"],
    "累": ["疲惫", "辛苦", "休息"],
    "加班": ["工作", "辛苦", "休息"],
    "工作": ["加班", "辛苦"],
    "邀约": ["见面", "周末", "可拒绝"],
}

SKILL_ID = "goutoujunshi"
SKILL_NAME = "狗头军师"
UPSTREAM_URL = "https://github.com/powerycy/goutoujunshi.git"
UPSTREAM_COMMIT = "81d99da388da11cb6c1bc18259c6802c82fbaf41"
SOURCE_ROOT = Path(__file__).with_name("resources") / "goutoujunshi_source"

_REFERENCE_ROUTES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("家暴", "胁迫", "跟踪", "威胁", "诈骗", "自伤", "伤人", "财务控制"), ("references/knowledge/17-中国法律安全与危机转介.md",)),
    (("同意", "性", "亲密", "避孕", "身体", "接触"), ("references/knowledge/08-同意边界性与亲密.md",)),
    (("pua", "推拉", "冷读", "服从", "煤气灯", "贬低", "mystery", "blueprint"), ("references/knowledge/05-PUA操控与伦理替代.md", "references/knowledge/20-经典社交体系的机制、证据与风险边界.md")),
    (("工资", "收入", "钱", "房", "彩礼", "家务", "育儿", "父母", "家庭条件"), ("references/knowledge/12-金钱家务育儿与双方家庭.md", "references/knowledge/06-吸引约会与关系启动.md")),
    (("分手", "复合", "背叛", "出轨", "离婚"), ("references/knowledge/15-分手背叛与关系修复.md", "references/practical/关系投入失衡：互惠判断、降级投入与退出决策.md")),
    (("吵架", "冲突", "道歉", "沟通", "误会"), ("references/knowledge/07-沟通冲突与修复.md", "references/practical/万能吵架技巧：理性冲突处理指南.md")),
    (("mbti", "人格", "依恋", "焦虑", "内耗", "情绪"), ("references/knowledge/03-依恋理论与情绪调节.md", "references/knowledge/04-MBTI人格与匹配.md")),
    (("不回", "冷淡", "累", "疲惫", "加班", "忙", "投入", "失衡", "断联", "退出"), ("references/practical/关系投入失衡：互惠判断、降级投入与退出决策.md", "references/knowledge/09-在线约会与数字关系.md")),
    (("邀约", "约会", "见面", "表白", "喜欢", "关系确认", "一直", "安全感"), ("references/knowledge/06-吸引约会与关系启动.md", "references/practical/主动表达、第一次见面与自然接触.md")),
    (("截图", "聊天记录", "网聊", "语音", "怎么回", "回复"), ("references/knowledge/09-在线约会与数字关系.md", "references/practical/实战话术编排器：从一句回复到后续分支.md")),
)
_DEFAULT_REFERENCES = (
    "references/knowledge/02-亲密关系心理学总论.md",
    "references/practical/实战话术编排器：从一句回复到后续分支.md",
)


class VectorIndexUnavailable(RuntimeError):
    """The semantic index is absent, stale, or cannot be safely used."""


class SourceRef(BaseModel):
    path: str = Field(min_length=1)
    sha256: str | None = None


class KnowledgeCard(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    scenario: str = Field(min_length=1, max_length=120)
    triggers: list[str] = Field(min_length=1, max_length=12)
    user_goal: str = Field(min_length=1, max_length=240)
    safe_principles: list[str] = Field(min_length=1, max_length=8)
    response_constraints: list[str] = Field(min_length=1, max_length=8)
    source_refs: list[SourceRef] = Field(min_length=1)
    source_risk: str
    review_status: str


class CardBatch(BaseModel):
    status: str = Field(pattern=r"^(approved|research_only)$")
    cards: list[KnowledgeCard] = Field(default_factory=list, max_length=4)
    note: str = Field(default="", max_length=300)


def read_cards(path: Path = CARDS_PATH) -> list[dict[str, Any]]:
    try:
        raw_cards = json.loads(path.read_text(encoding="utf-8"))
        if GENERATED_CARDS_PATH.exists() and path == CARDS_PATH:
            raw_cards += json.loads(GENERATED_CARDS_PATH.read_text(encoding="utf-8")).get("cards", [])
        cards = [KnowledgeCard.model_validate(card) for card in raw_cards]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as error:
        raise ValueError("知识卡文件无效") from error
    approved = []
    raw_root = RAW_KNOWLEDGE_DIR.resolve()
    raw_corpus_available = any(RAW_KNOWLEDGE_DIR.rglob("*.md"))
    for card in cards:
        text = card_text(card.model_dump())
        if card.review_status != "approved" or any(word in text for word in CARD_UNSAFE_PATTERN):
            continue
        for ref in card.source_refs:
            source = KNOWLEDGE_DIR / ref.path
            if raw_root not in source.resolve().parents:
                raise ValueError(f"知识卡 {card.id} 的溯源文件无效")
            if not source.is_file():
                if raw_corpus_available:
                    raise ValueError(f"知识卡 {card.id} 的溯源文件无效")
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if ref.sha256 and ref.sha256 != digest:
                raise ValueError(f"知识卡 {card.id} 的溯源文件已变化，需要重新审核")
        approved.append(card.model_dump())
    return approved


def card_text(card: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"场景：{card['scenario']}",
            f"触发词：{'、'.join(card['triggers'])}",
            f"用户目标：{card['user_goal']}",
            f"安全原则：{'；'.join(card['safe_principles'])}",
            f"回复约束：{'；'.join(card['response_constraints'])}",
        ]
    )


def safe_card_index() -> list[dict[str, str]]:
    return [
        {
            "source": f"card:{card['id']}",
            "title": card["scenario"],
            "content": card_text(card),
            "tier": "safe",
        }
        for card in read_cards()
    ]


SAFE_CARD_IDS = frozenset(card["id"] for card in read_cards())
SAFE_CARD_INDEX = safe_card_index()


def expand_keywords(keywords: list[str]) -> list[str]:
    expanded = [term.strip() for term in keywords if term.strip()]
    for term in list(expanded):
        for key, aliases in TERM_ALIASES.items():
            if key in term or term in key:
                expanded.extend(aliases)
    return list(dict.fromkeys(expanded))[:10]


def retrieve_by_keywords(
    keywords: list[str],
    index: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    terms = expand_keywords(keywords)
    scored: list[tuple[int, dict[str, str]]] = []
    for chunk in SAFE_CARD_INDEX if index is None else index:
        if chunk["tier"] != "safe":
            continue
        haystack = chunk["content"].lower()
        heading = f"{chunk['title']} {chunk['source']}".lower()
        score = sum(
            haystack.count(term.lower()) + heading.count(term.lower()) * 4
            for term in terms
        )
        if score:
            scored.append((score, chunk))
    return [
        chunk
        for _, chunk in sorted(
            scored,
            key=lambda item: (-item[0], item[1]["source"]),
        )[:4]
    ]


def selected_reference_paths(context: str, limit: int = 3) -> list[Path]:
    normalized = context.lower()
    selected: list[str] = []
    for triggers, paths in _REFERENCE_ROUTES:
        if any(trigger in normalized for trigger in triggers):
            selected.extend(paths)
        if len(dict.fromkeys(selected)) >= limit:
            break
    if not selected:
        selected.extend(_DEFAULT_REFERENCES)
    return [SOURCE_ROOT / path for path in dict.fromkeys(selected)][:limit]


def reference_context(context: str, limit: int = 3) -> str:
    documents = []
    for path in selected_reference_paths(context, limit):
        if path.is_file():
            relative_path = path.relative_to(SOURCE_ROOT).as_posix()
            documents.append(
                f"<REFERENCE path='{relative_path}'>\n"
                f"{path.read_text(encoding='utf-8')}\n"
                "</REFERENCE>"
            )
    return "\n\n".join(documents)


def research_manifest(root: Path = RAW_KNOWLEDGE_DIR) -> list[dict[str, str]]:
    """Return one immutable provenance record per raw document without copying it."""
    return [
        {
            "path": path.relative_to(KNOWLEDGE_DIR).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "research_only",
        }
        for path in sorted(root.rglob("*.md"))
    ]


def write_manifest(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(research_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_research_cards(output: Path = RESEARCH_CARDS_PATH) -> list[dict[str, Any]]:
    """Create one non-chat research card per raw document, preserving provenance."""
    previous = {}
    if output.exists():
        previous = {item["path"]: item for item in json.loads(output.read_text(encoding="utf-8")).get("cards", [])}
    cards = []
    for source in research_manifest():
        prior = previous.get(source["path"], {})
        cards.append({
            "id": f"research-{source['sha256'][:16]}",
            "path": source["path"],
            "sha256": source["sha256"],
            "status": prior.get("status", "research_only"),
            "note": prior.get("note", "原始资料已纳入仅研究用途的向量索引；不得用于在线聊天生成。"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"version": 1, "cards": cards}, ensure_ascii=False, indent=2), encoding="utf-8")
    return cards


def build_research_vector_index(output: Path = RESEARCH_VECTOR_INDEX_PATH, encode_fn=None, cards_path: Path = RESEARCH_CARDS_PATH) -> dict[str, Any]:
    """Embed every raw document for offline research only; raw text is not persisted."""
    cards = build_research_cards(cards_path)
    texts = [(KNOWLEDGE_DIR / card["path"]).read_text(encoding="utf-8") for card in cards]
    encode_fn = encode_fn or embed
    vectors = encode_fn(texts)
    if len(vectors) != len(cards):
        raise ValueError("研究向量数量不匹配")
    payload = {"version": 1, "embedding_model": EMBEDDING_MODEL, "scope": "research_only", "cards": [{"id": card["id"], "path": card["path"], "sha256": card["sha256"], "vector": vector} for card, vector in zip(cards, vectors)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def build_cards(output: Path = GENERATED_CARDS_PATH) -> dict[str, Any]:
    """Safely transform every raw document offline, with per-document resume data.

    This is deliberately not part of the request path: raw text is supplied only
    to the offline transformer, while chat later reads only validated cards.
    """
    prior = {item["path"]: item for item in json.loads(output.read_text(encoding="utf-8")).get("sources", [])} if output.exists() else {}
    records, cards = [], []
    model_client = create_model_client(SETTINGS)
    prompt = (
        SYSTEM_PROMPT
        + "\n你在离线整理研究材料。将下列单篇材料转换为 0-4 张安全、平等、尊重边界的知识卡。"
        "不得复述操控、贬损、性越界或施压话术；无法安全改写时 status=research_only、cards=[]。"
        "每张卡的 source_refs 必须给一个非空 path 占位，程序会校正为真实溯源。JSON 字段：status,cards,note。"
    )
    for source in research_manifest():
        cached = prior.get(source["path"])
        if cached and cached.get("sha256") == source["sha256"] and cached.get("status") in {"approved", "research_only"}:
            records.append(cached)
            cards.extend(cached.get("cards", []))
            continue
        raw = (KNOWLEDGE_DIR / source["path"]).read_text(encoding="utf-8")
        try:
            batch = model_client.call_json(
                prompt,
                f"<SOURCE_PATH>{source['path']}</SOURCE_PATH>\n"
                f"<RAW_RESEARCH>{raw}</RAW_RESEARCH>",
                CardBatch,
            )
            generated = []
            for card in batch.cards:
                value = card.model_dump()
                value["id"] = f"{Path(source['path']).stem}-{source['sha256'][:8]}-{len(generated) + 1}".lower().replace("_", "-")
                value["source_refs"] = [{"path": source["path"], "sha256": source["sha256"]}]
                value["source_risk"] = "mixed"
                value["review_status"] = batch.status
                generated.append(value)
            record = {**source, "status": batch.status, "cards": generated, "note": batch.note}
        except Exception as error:
            record = {**source, "status": "failed", "cards": [], "note": type(error).__name__}
        records.append(record)
        cards.extend(record["cards"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"version": 1, "sources": records, "cards": cards}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sources": records, "cards": cards}


@lru_cache(maxsize=1)
def embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # keeps chat usable until semantic search is installed
        raise RuntimeError("未安装 sentence-transformers；请在 backend 执行 uv sync") from error
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return embedding_model().encode(texts, normalize_embeddings=True).tolist()


def build_vector_index(output: Path = VECTOR_INDEX_PATH, encode_fn=embed) -> dict[str, Any]:
    cards = read_cards()
    payload = {
        "version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "cards": [{"id": card["id"], "text": card_text(card), "vector": vector} for card, vector in zip(cards, encode_fn([card_text(card) for card in cards]))],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / max(math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)), 1e-12)


def retrieve_by_vector(query: str, limit: int = 4, index_path: Path = VECTOR_INDEX_PATH) -> list[dict[str, Any]]:
    if not query.strip() or not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("version") != 1 or index.get("embedding_model") != EMBEDDING_MODEL:
            raise VectorIndexUnavailable("向量索引版本或模型不匹配")
        cards = {card["id"]: card for card in read_cards()}
        items = index.get("cards")
        if not isinstance(items, list):
            raise VectorIndexUnavailable("向量索引缺少卡片")
        query_vector = embed([query])[0]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in query_vector):
            raise VectorIndexUnavailable("查询向量无效")
        ranked = []
        for item in items:
            vector = item.get("vector") if isinstance(item, dict) else None
            if not isinstance(item, dict) or item.get("id") not in cards or not isinstance(vector, list) or len(vector) != len(query_vector) or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
                raise VectorIndexUnavailable("向量索引内容无效")
            ranked.append((cosine(query_vector, vector), cards[item["id"]]))
        return [card for score, card in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit] if score >= 0.3]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise VectorIndexUnavailable("向量检索不可用") from error


def knowledge_cli() -> None:
    parser = argparse.ArgumentParser(description="CrushPilot safe knowledge-card maintenance")
    parser.add_argument("command", choices=("manifest", "build-cards", "build-index", "build-research-index"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        output = args.output or DATA_DIR / "research_manifest.json"
        write_manifest(output)
        print(f"wrote {len(research_manifest())} raw-document provenance records to {output}")
    elif args.command == "build-cards":
        output = args.output or GENERATED_CARDS_PATH
        result = build_cards(output)
        print(f"processed {len(result['sources'])} raw documents and wrote {len(result['cards'])} derived cards to {output}")
    elif args.command == "build-research-index":
        output = args.output or RESEARCH_VECTOR_INDEX_PATH
        print(f"wrote {len(build_research_vector_index(output)['cards'])} research-only vectors to {output}")
    else:
        output = args.output or VECTOR_INDEX_PATH
        print(f"wrote {len(build_vector_index(output)['cards'])} safe-card vectors to {output}")


if __name__ == "__main__":
    knowledge_cli()
