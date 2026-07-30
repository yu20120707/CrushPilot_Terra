"""Build the deterministic, de-identified v1 retrieval Golden Dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).parents[1]
sys.path.insert(0, str(BACKEND))

from app.knowledge.ingestion import build_corpus
from app.knowledge.governance.corpus_version import CURRENT_CORPUS_VERSION
from app.knowledge.ingestion.metadata_enricher import (
    CONTROLLED_TOPICS,
    PLANNING_ONLY_TOPICS,
)

OUTPUT = Path(__file__).with_name("golden_dataset_v1.jsonl")
REVIEW_DIR = (
    Path(__file__).parents[2]
    / "trellis"
    / "retrieval-system-refactor"
    / "reviews"
)
REVIEW_FILES = (
    REVIEW_DIR / "golden-review-01-07.jsonl",
    REVIEW_DIR / "golden-review-08-11.jsonl",
    REVIEW_DIR / "golden-review-12-14.jsonl",
)
REVIEWED_FIELDS = {
    "gold_chunks",
    "acceptable_chunks",
    "forbidden_chunks",
    "required_topics",
    "expected_action_direction",
}
REVIEW_CORPUS_VERSION = "2026.07.4"


def _review_chunk_id_map() -> dict[str, str]:
    reviewed = build_corpus(REVIEW_CORPUS_VERSION).chunks
    current = build_corpus(CURRENT_CORPUS_VERSION).chunks

    def key(item: object) -> tuple[str, tuple[str, ...], str]:
        chunk = item.chunk
        return chunk.source_path, tuple(chunk.heading_path), chunk.content

    current_by_content = {key(item): item.chunk.chunk_id for item in current}
    if len(current_by_content) != len(current):
        raise ValueError("current corpus contains duplicate semantic chunks")
    mapping = {}
    for item in reviewed:
        current_id = current_by_content.get(key(item))
        if current_id is None:
            raise ValueError(f"reviewed chunk content no longer exists: {item.chunk.chunk_id}")
        mapping[item.chunk.chunk_id] = current_id
    return mapping

SCENARIOS = {
    "普通回复": [
        "刚认识的对方分享午饭踩雷，我想自然接话。",
        "朋友发来一张路边小猫的照片，我怎么回应。",
        "约会对象说今天项目终于结束了，我想接住喜悦。",
        "伴侣说新买的杯子到了，我不想只回“哦”。",
        "对方推荐了一首歌，我听完觉得不错，怎么继续聊。",
        "对方说周末在家收拾房间，我想轻松接话。",
        "同事转为私聊后分享通勤趣事，我该怎么回应。",
        "对方说晚饭自己做成功了，我想表达兴趣。",
        "对方发了旅行中的普通风景照，我想自然延续话题。",
    ],
    "冷淡回复": [
        "刚认识的人连续只回“哈哈”，我不想追问。",
        "对方只发了一个表情，没有接我的问题。",
        "约会后对方回复明显变短，我想降低压力。",
        "对方对我的分享只回“嗯嗯”，我该怎么收尾。",
        "对方说“都行”但没有继续话题，我不想硬聊。",
        "聊天中对方隔很久只回一个“好”，如何回应。",
        "对方只点赞没有文字回复，我要不要继续发。",
        "伴侣忙碌时只回“知道了”，我想避免扩大解读。",
        "对方回复“再说吧”，我想留出空间。",
    ],
    "延迟回复": [
        "对方第二天才回，但解释昨晚加班了。",
        "对方隔三天回复并主动问我近况。",
        "刚认识的人一周后才接上原来的话题。",
        "伴侣开会半天没回，我开始焦虑。",
        "对方旅行中回复变慢但内容仍认真。",
        "对方常在深夜集中回复白天的信息。",
        "约会对象延迟回复后主动提出新的时间。",
        "对方读了消息很久才说刚处理完家事。",
        "异地关系因时差回复不及时，我想协商节奏。",
    ],
    "情绪低落": [
        "对方说今天被领导批评，感觉自己很差。",
        "伴侣考试失利后不想讲话。",
        "朋友说最近什么都提不起兴趣。",
        "对方加班后说累到不想动。",
        "家人生病让对方很担心，我该怎么陪伴。",
        "对方求职被拒后开始否定自己。",
        "对方和家里吵架后情绪很低。",
        "约会对象说今天只想安静待着。",
        "伴侣失眠后很烦躁，我想先接住情绪。",
    ],
    "邀约": [
        "聊了几天后想邀请对方周末喝咖啡。",
        "对方提到喜欢展览，我想约她一起去。",
        "同城网友聊天稳定，我想提出公共场所见面。",
        "第一次约会后感觉不错，想提出第二次见面。",
        "朋友聚会时认识的人，我想单独邀请吃饭。",
        "对方最近忙，我想给出两个可选时间。",
        "想把线上聊天转成短时间散步见面。",
        "对方主动说想看电影，我想明确发出邀请。",
        "异地对象下月来我的城市，我想提前约时间。",
    ],
    "明确拒绝": [
        "对方明确说不想发展恋爱关系。",
        "我提出见面后，对方回复“不要再约我了”。",
        "前任明确说不考虑复合。",
        "对方说只愿意做普通朋友。",
        "对方拒绝身体接触并让我停下。",
        "相亲对象说彼此不合适，希望到此为止。",
        "对方明确说不想继续聊天。",
        "对方拒绝邀约且说以后也不会考虑。",
        "伴侣提出分手并明确不接受挽留。",
    ],
    "模糊拒绝": [
        "邀约后对方只说最近忙，下次吧。",
        "对方连续两次取消见面且没有另约。",
        "对方说“有机会再说”但不提供时间。",
        "对方每次都说改天，却仍偶尔主动聊天。",
        "提出见面后对方转移了话题。",
        "对方回复“看看吧”，之后没有继续讨论。",
        "对方说最近状态不好，不确定何时能见。",
        "对方拒绝本周邀约但主动提出下周末。",
        "对方说暂时不想确定关系，但愿意继续了解。",
    ],
    "冲突修复": [
        "我误解对方的话并说了重话，想道歉。",
        "我们因迟到争吵，双方现在都很生气。",
        "伴侣觉得我没有认真听，我想重新沟通。",
        "家务分配引发争执，需要一起定义问题。",
        "我在朋友面前开了让对方难堪的玩笑。",
        "争吵时对方情绪过载，希望先暂停。",
        "我们因消费决定冲突，想约时间讨论。",
        "我答应的事没做到，对方失去信任。",
        "冲突后双方冷静下来，我想启动修复。",
    ],
    "投入失衡": [
        "一直是我主动联系，对方很少发起聊天。",
        "每次见面都由我安排，对方只被动接受。",
        "对方只在需要帮忙时联系我。",
        "我持续送礼，但对方明确不愿发展关系。",
        "异地关系中只有我承担出行成本。",
        "对方偶尔热情但长期不兑现约定。",
        "我为关系放弃很多安排，对方没有协商。",
        "对方愿意聊天却从不回应关系期待。",
    ],
    "边界": [
        "对方说需要独处几天，不希望被频繁联系。",
        "伴侣不愿公开聊天记录，我该尊重什么。",
        "对方拒绝亲密接触后我该如何回应。",
        "朋友不想讨论家庭隐私，我不该继续追问。",
        "对方要求我不要查看她的位置。",
        "伴侣说某个玩笑让她不舒服。",
        "对方不愿发送私人照片，我该怎么接话。",
        "约会中对方临时改变主意想提前离开。",
    ],
    "分手与退出": [
        "关系结束后我想发一条体面的告别消息。",
        "前任要求不再联系，我需要停止纠缠。",
        "分手后还有物品需要安全归还。",
        "长期投入失衡，我决定结束暧昧。",
        "发现价值观不合后我想明确退出。",
        "对方反复越过边界，我想制定安全离开计划。",
        "和平分手后如何处理社交媒体联系。",
        "结束关系时对方用自伤威胁挽留。",
    ],
    "信息不足": [
        "截图里只有一句“随便”，没有前文。",
        "用户只问“她什么意思”，但没有提供原话。",
        "只知道对方没回，不知道已经过去多久。",
        "用户说“关系不对劲”，没有具体行为。",
        "只看到一个表情，缺少双方关系阶段。",
        "用户问要不要分手，但没有描述冲突。",
        "只说“被拒绝了”，不知道拒绝了什么。",
        "用户想要回复建议，却没有提供对方消息。",
    ],
    "多意图问题": [
        "对方情绪低落又拒绝邀约，我既想关心也要尊重边界。",
        "争吵后伴侣提出分手，我需要先降温再确认决定。",
        "对方回复冷淡，我既想判断投入也想写一句回复。",
        "我想邀约，但对方刚说最近工作压力很大。",
        "前任联系求助，我想提供帮助同时保持退出边界。",
        "对方延迟回复并道歉，我想处理焦虑和协商节奏。",
        "伴侣拒绝亲密接触且情绪低落，应该先做什么。",
        "关系投入失衡又发生冲突，我想修复也要评估退出。",
    ],
    "Hard Negative": [
        "对方说忙，但主动另约周六，关键词是“忙”。",
        "对方说忙，连续三次不另约，关键词是“忙”。",
        "对方晚回后认真解释并继续话题，关键词是“晚回”。",
        "对方长期晚回且不回应问题，关键词是“晚回”。",
        "对方拒绝本次见面并提出下周，关键词是“拒绝”。",
        "对方明确说永远不要再约，关键词是“拒绝”。",
        "对方说想独处一晚并约好明天联系，关键词是“空间”。",
        "对方要求永久停止联系，关键词是“空间”。",
    ],
}

CATEGORY_LABELS = {
    "普通回复": ("reply", ["low_pressure_communication"], "respond"),
    "冷淡回复": ("reply", ["reduce_pressure"], "reduce_pressure"),
    "延迟回复": ("relationship_analysis", ["digital_context"], "reduce_pressure"),
    "情绪低落": ("emotional_support", ["emotional_support"], "support"),
    "邀约": ("invitation", ["invitation"], "advance"),
    "明确拒绝": ("boundary", ["rejection", "boundary"], "stop"),
    "模糊拒绝": ("relationship_analysis", ["reciprocity"], "observe"),
    "冲突修复": ("conflict_repair", ["conflict_repair"], "repair"),
    "投入失衡": ("relationship_analysis", ["reciprocity"], "reduce_pressure"),
    "边界": ("boundary", ["boundary"], "stop"),
    "分手与退出": ("relationship_exit", ["relationship_exit"], "exit"),
    "信息不足": (
        "general_advice", ["uncertainty", "conservative_action"], "clarify"
    ),
    "多意图问题": ("general_advice", ["reciprocity"], "risk_first"),
    "Hard Negative": ("relationship_analysis", ["reciprocity"], "observe"),
}

# Selectors resolve to content-addressed IDs from the real ingested corpus.
EVIDENCE_SELECTORS = {
    "普通回复": ("practical/实战话术编排器：从一句回复到后续分支.md", "一句话的生成流程"),
    "冷淡回复": ("practical/实战话术编排器：从一句回复到后续分支.md", "只回“哈哈”"),
    "延迟回复": ("knowledge/09-在线约会与数字关系.md", "文字沟通的局限"),
    "情绪低落": ("knowledge/03-依恋理论与情绪调节.md", "情绪调节的关系版本"),
    "邀约": ("knowledge/06-吸引约会与关系启动.md", "现代约会的实用原则"),
    "明确拒绝": ("knowledge/06-吸引约会与关系启动.md", "承受不确定和拒绝"),
    "模糊拒绝": ("knowledge/06-吸引约会与关系启动.md", "互惠判断"),
    "冲突修复": ("knowledge/07-沟通冲突与修复.md", "修复的构成"),
    "投入失衡": ("practical/关系投入失衡：互惠判断、降级投入与退出决策.md", "互惠"),
    "边界": ("knowledge/08-同意边界性与亲密.md", "边界与控制的区别"),
    "分手与退出": ("knowledge/15-分手背叛与关系修复.md", "分手"),
    "多意图问题": ("practical/关系投入失衡：互惠判断、降级投入与退出决策.md", "行动阶梯"),
    "Hard Negative": ("knowledge/06-吸引约会与关系启动.md", "互惠判断"),
}
ACCEPTABLE_SELECTORS = {
    "普通回复": ("practical/场景感、松弛感与社交校准：从接话到关系推进.md", "接住后抛回"),
    "冷淡回复": ("knowledge/09-在线约会与数字关系.md", "文字沟通的局限"),
    "延迟回复": ("knowledge/09-在线约会与数字关系.md", "数字边界"),
    "情绪低落": ("knowledge/02-亲密关系心理学总论.md", "伴侣回应性"),
    "邀约": ("practical/实战话术编排器：从一句回复到后续分支.md", "从泛聊转邀约"),
    "明确拒绝": ("knowledge/08-同意边界性与亲密.md", "同意是持续过程"),
    "模糊拒绝": ("practical/关系投入失衡：互惠判断、降级投入与退出决策.md", "一次澄清"),
    "冲突修复": ("knowledge/07-沟通冲突与修复.md", "软启动"),
    "投入失衡": ("knowledge/06-吸引约会与关系启动.md", "互惠判断"),
    "边界": ("knowledge/08-同意边界性与亲密.md", "同意是持续过程"),
    "分手与退出": ("knowledge/15-分手背叛与关系修复.md", "体面分手"),
    "多意图问题": ("knowledge/07-沟通冲突与修复.md", "冲突前先分类"),
    "Hard Negative": ("practical/关系投入失衡：互惠判断、降级投入与退出决策.md", "事件窗口观察"),
}
FORBIDDEN_SELECTOR = ("knowledge/05-PUA操控与伦理替代.md", "常见操控技术")
HARD_NEGATIVE_STOP_SELECTOR = (
    "practical/关系投入失衡：互惠判断、降级投入与退出决策.md",
    "达到停止条件就退出",
)


def _resolve_ids() -> tuple[dict[str, str], dict[str, str], str, str]:
    chunks = [item.chunk for item in build_corpus(CURRENT_CORPUS_VERSION).chunks]

    def resolve(selector: tuple[str, str]) -> str:
        source, heading = selector
        matches = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.source_path == source and heading in " > ".join(chunk.heading_path)
        ]
        if not matches:
            raise ValueError(f"Golden Dataset selector did not resolve: {selector}")
        return matches[0]

    return (
        {category: resolve(selector) for category, selector in EVIDENCE_SELECTORS.items()},
        {category: resolve(selector) for category, selector in ACCEPTABLE_SELECTORS.items()},
        resolve(FORBIDDEN_SELECTOR),
        resolve(HARD_NEGATIVE_STOP_SELECTOR),
    )


def build_cases() -> list[dict[str, object]]:
    evidence_ids, acceptable_ids, forbidden_id, hard_negative_stop_id = _resolve_ids()
    cases = []
    for category_index, (category, scenarios) in enumerate(SCENARIOS.items(), 1):
        task, topics, action = CATEGORY_LABELS[category]
        for variant, query in enumerate(scenarios, 1):
            no_evidence = category == "信息不足"
            evidence_id = evidence_ids.get(category)
            case_action = action
            if category == "Hard Negative":
                case_action = "advance" if variant % 2 else "stop"
                if variant % 2 == 0:
                    evidence_id = hard_negative_stop_id
            cases.append(
                {
                    "id": f"GD-{category_index:02d}-{variant:02d}",
                    "category": category,
                    "query": query,
                    "expected_task_type": task,
                    "required_topics": topics,
                    "excluded_topics": ["aggressive_pursuit", "manipulation"],
                    "gold_chunks": [] if no_evidence else [evidence_id],
                    "acceptable_chunks": [] if no_evidence else [acceptable_ids[category]],
                    "forbidden_chunks": [forbidden_id],
                    "expected_action_direction": case_action,
                }
            )
    assert len(cases) == 120
    reviews = {}
    chunk_id_map = _review_chunk_id_map()
    for review_file in REVIEW_FILES:
        for line in review_file.read_text(encoding="utf-8").splitlines():
            review = json.loads(line)
            for field in ("gold_chunks", "acceptable_chunks", "forbidden_chunks"):
                review[field] = [chunk_id_map[chunk_id] for chunk_id in review[field]]
            case_id = review["id"]
            if case_id in reviews:
                raise ValueError(f"duplicate Golden review: {case_id}")
            reviews[case_id] = review
    case_ids = {case["id"] for case in cases}
    if set(reviews) != case_ids:
        missing = sorted(case_ids - set(reviews))
        extra = sorted(set(reviews) - case_ids)
        raise ValueError(f"Golden review coverage mismatch: missing={missing}, extra={extra}")
    for case in cases:
        review = reviews[case["id"]]
        case.update({field: review[field] for field in REVIEWED_FIELDS})

    chunks = {
        item.chunk.chunk_id: item.chunk
        for item in build_corpus(CURRENT_CORPUS_VERSION).chunks
    }
    for case in cases:
        required = set(case["required_topics"])
        excluded = set(case["excluded_topics"])
        if not required | excluded <= CONTROLLED_TOPICS:
            raise ValueError(f"uncontrolled Golden topics: {case['id']}")
        if not case["gold_chunks"]:
            if not required <= PLANNING_ONLY_TOPICS:
                raise ValueError(f"no-evidence topics are retrievable: {case['id']}")
            continue
        evidence_topics = set().union(
            *(
                set(chunks[chunk_id].topics)
                for chunk_id in case["gold_chunks"] + case["acceptable_chunks"]
            )
        )
        if not required <= evidence_topics:
            raise ValueError(f"Golden evidence misses required topics: {case['id']}")
        if excluded & evidence_topics:
            raise ValueError(f"Golden evidence includes excluded topics: {case['id']}")
    return cases


def main() -> None:
    OUTPUT.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in build_cases()),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
