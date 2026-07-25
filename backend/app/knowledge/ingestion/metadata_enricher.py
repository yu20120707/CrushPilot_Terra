from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuggestedMetadata:
    knowledge_type: str
    topics: list[str]
    action_labels: list[str]
    applicable_when: list[str]
    not_applicable_when: list[str]


TOPIC_RULES = {
    "rejection": ("拒绝", "婉拒", "不见面"),
    "boundary": ("边界", "同意", "越界", "施压"),
    "reduce_pressure": (
        "降压", "留空间", "不施压", "降低压力", "只回", "不追问", "收尾"
    ),
    "emotional_support": ("情绪", "安慰", "倾听", "共情", "情绪价值"),
    "low_pressure_communication": ("松弛", "低压力", "自然沟通"),
    "invitation": ("邀约", "约会", "见面"),
    "conflict_repair": ("冲突", "吵架", "修复", "道歉"),
    "relationship_exit": ("分手", "退出", "背叛"),
    "reciprocity": ("互惠", "投入", "失衡"),
    "digital_context": ("延迟回复", "晚回", "文字沟通", "数字边界", "线上互动"),
    "manipulation": ("操控", "pua", "欺骗", "强迫"),
}

PLANNING_ONLY_TOPICS = {
    "aggressive_pursuit",
    "conservative_action",
    "forced_disclosure",
    "mind_reading",
    "relationship_escalation",
    "uncertainty",
}
CONTROLLED_TOPICS = (
    frozenset(TOPIC_RULES)
    | PLANNING_ONLY_TOPICS
    | {"general_relationship_advice"}
)


def suggest_metadata(title: str, heading_path: list[str], content: str) -> SuggestedMetadata:
    text = "\n".join((title, *heading_path, content))
    lowered = text.lower()
    structural_text = "\n".join((title, *heading_path)).lower()
    topics = [
        topic
        for topic, keywords in TOPIC_RULES.items()
        if any(
            keyword.lower()
            in (structural_text if topic == "manipulation" else lowered)
            for keyword in keywords
        )
    ] or ["general_relationship_advice"]
    if any(keyword in text for keyword in ("禁止", "不得", "风险", "红线", "危机")):
        knowledge_type = "hard_rule"
    elif any(keyword in text for keyword in ("案例", "示例", "对话", "话术")):
        knowledge_type = "example"
    elif any(keyword in text for keyword in ("步骤", "方法", "策略", "实践", "指南")):
        knowledge_type = "strategy"
    else:
        knowledge_type = "principle"

    actions: list[str] = []
    not_applicable: list[str] = []
    if set(topics) & {"rejection", "boundary", "reduce_pressure", "relationship_exit"}:
        actions.extend(("reduce_pressure", "stop"))
    if "invitation" in topics and not set(topics) & {"rejection", "boundary"}:
        actions.append("advance")
        not_applicable.extend(("explicit_rejection", "explicit_boundary"))
    if "manipulation" in topics:
        actions.append("stop")
        not_applicable.append("normal_online_advice")
    if "emotional_support" in topics:
        actions.append("support")
    return SuggestedMetadata(
        knowledge_type=knowledge_type,
        topics=list(dict.fromkeys(topics)),
        action_labels=list(dict.fromkeys(actions)),
        applicable_when=list(dict.fromkeys(topics)),
        not_applicable_when=list(dict.fromkeys(not_applicable)),
    )
