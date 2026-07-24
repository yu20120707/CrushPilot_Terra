from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.agents.assistant import tools
from app.agents.assistant.prompts import (
    CHAT_SYSTEM_PROMPT,
    FINAL_RESPONSE_INSTRUCTION,
    INTENT_PROMPT,
)
from app.agents.assistant.schemas import (
    ChatResult,
    GeneratedChatResult,
    IntentAnalysis,
)
from app.agents.assistant.state import AssistantState
from app.core.config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)

HIGH_RISK_FLAGS = frozenset({"unsafe_request"})
INPUT_UNSAFE_PATTERN = re.compile(
    r"跟踪|尾随|蹲守|堵(?:她|他|人|门|路)|威胁|恐吓|强迫|逼迫|"
    r"纠缠|骚扰|未成年|骗(?:她|他)|诈骗|冒充|假装身份|灌醉|"
    r"下药|偷拍|常去.{0,8}(?:等|堵)|连续.{0,8}(?:联系|发消息)"
)
DANGEROUS_ADVICE_PATTERN = re.compile(
    r"跟踪|尾随|蹲守|威胁|恐吓|强迫|逼迫|骚扰|诈骗|冒充|"
    r"灌醉|下药|偷拍|偷拍视频|假装(?:偶遇|身份)|"
    r"(?:公司|单位|家|学校).{0,8}(?:楼下|门口).{0,8}(?:等|堵)|"
    r"反复.{0,10}(?:发消息|联系)"
)
NEGATED_ADVICE_PREFIX = re.compile(
    r"(?:不要|别(?:再|去|用|想)|不应|避免|拒绝|禁止|不能|不可以).{0,10}$"
)


class ModelClient(Protocol):
    def call_json(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
    ) -> SchemaT: ...


def contains_unsafe_input(text: str) -> bool:
    return bool(INPUT_UNSAFE_PATTERN.search(text))


def contains_unsafe_advice(text: str) -> bool:
    return any(
        not NEGATED_ADVICE_PREFIX.search(text[max(0, match.start() - 16) : match.start()])
        for match in DANGEROUS_ADVICE_PATTERN.finditer(text)
    )


def safe_result(intent: str = "边界风险") -> ChatResult:
    return ChatResult(
        intent=intent,
        judgement="狗头军师判断：这类做法可能伤害对方或越过边界。",
        recommended_reply="先尊重对方的意愿和边界，不要继续施压。",
        alternatives=[
            "如果对方不想继续，请给彼此一点空间。",
            "先冷静下来，再用尊重的方式沟通。",
        ],
        warning=(
            "不提供操控、骚扰、威胁、跟踪、欺骗、"
            "性越界或未成年人相关建议。"
        ),
    )


def validate_result(result: ChatResult, intent: str) -> ChatResult:
    candidate_text = "\n".join([result.recommended_reply, *result.alternatives])
    if contains_unsafe_advice(candidate_text):
        return safe_result(intent)
    return result.model_copy(update={"intent": intent, "warning": None})


def demo_analysis(message: str) -> IntentAnalysis:
    terms = [
        term
        for term in re.findall(r"[\u4e00-\u9fff]{2,8}", message)
        if term not in {"怎么回复", "我怎么回"}
    ]
    return IntentAnalysis(
        scene="聊天回应",
        goal="给出自然、尊重边界的回应",
        features=["需要回应对方", "需要控制表达压力"],
        keywords=(terms or ["沟通", "尊重"])[:6],
        risk_flags=[],
    )


def evidence_text(chunks: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"来源：{chunk['source']}\n内容：{chunk['content']}" for chunk in chunks
    )[:3200]


class AssistantNodes:
    """Dependency-injected LangGraph node collection."""

    def __init__(self, settings: Settings, model_client: ModelClient):
        self.settings = settings
        self.model_client = model_client

    def analyze_intent(self, state: AssistantState) -> AssistantState:
        message = state["user_message"]
        if contains_unsafe_input(message):
            analysis = IntentAnalysis(
                scene="边界风险",
                goal="拒绝越界请求并提供安全替代",
                features=["存在安全风险"],
                keywords=["尊重", "边界"],
                risk_flags=["unsafe_request"],
            )
        elif self.settings.demo_mode:
            analysis = demo_analysis(message)
        else:
            analysis = self.model_client.call_json(
                INTENT_PROMPT,
                f"<USER_MESSAGE>{message}</USER_MESSAGE>",
                IntentAnalysis,
            )
            analysis = analysis.model_copy(
                update={
                    "risk_flags": [
                        flag
                        for flag in analysis.risk_flags
                        if flag in HIGH_RISK_FLAGS
                    ]
                }
            )
        return {
            "intent_analysis": analysis.model_dump(),
            "retrieval_queries": analysis.keywords,
            "retrieval_attempt": 0,
        }

    def retrieve_knowledge(self, state: AssistantState) -> AssistantState:
        keyword_hits = tools.retrieve_by_keywords(
            state.get("retrieval_queries", [])
        )
        try:
            vector_hits = [
                {
                    "source": f"card:{card['id']}",
                    "title": card["scenario"],
                    "content": tools.card_text(card),
                    "tier": "safe",
                }
                for card in tools.retrieve_by_vector(state["user_message"])
            ]
        except (tools.VectorIndexUnavailable, RuntimeError):
            vector_hits = []
        merged = {
            chunk["source"]: chunk for chunk in [*vector_hits, *keyword_hits]
        }
        return {"retrieved_knowledge": list(merged.values())[:4]}

    def refine_query(self, state: AssistantState) -> AssistantState:
        return {
            "retrieval_queries": tools.expand_keywords(
                state.get("retrieval_queries", [])
            ),
            "retrieval_attempt": 1,
        }

    def generate_reply(self, state: AssistantState) -> AssistantState:
        analysis = IntentAnalysis.model_validate(state["intent_analysis"])
        if HIGH_RISK_FLAGS.intersection(analysis.risk_flags):
            result = safe_result(analysis.scene)
        elif self.settings.demo_mode:
            result = ChatResult(
                intent=analysis.scene,
                judgement="狗头军师建议先接住当下情绪，再给对方留出空间。",
                recommended_reply="听起来你今天挺累的，先好好休息，等你有空再聊。",
                alternatives=[
                    "辛苦啦，先让自己放松一下。",
                    "不用急着回复，忙完再说。",
                ],
            )
        else:
            route_context = "\n".join(
                [
                    state["user_message"],
                    analysis.scene,
                    analysis.goal,
                    *analysis.features,
                    *analysis.keywords,
                ]
            )
            prompt = (
                f"<INTENT>{analysis.model_dump_json()}</INTENT>\n"
                f"<EVIDENCE>{evidence_text(state.get('retrieved_knowledge', []))}"
                "</EVIDENCE>\n"
                "<GOUTOUJUNSHI_REFERENCES>"
                f"{tools.reference_context(route_context)}"
                "</GOUTOUJUNSHI_REFERENCES>\n"
                "<RECENT_DIALOGUE>"
                f"{json.dumps(state.get('messages', [])[-12:], ensure_ascii=False)}"
                "</RECENT_DIALOGUE>\n"
                f"<USER_MESSAGE>{state['user_message']}</USER_MESSAGE>\n"
                f"{FINAL_RESPONSE_INSTRUCTION}"
            )
            generated = self.model_client.call_json(
                CHAT_SYSTEM_PROMPT,
                prompt,
                GeneratedChatResult,
            )
            result = validate_result(
                ChatResult(
                    intent=analysis.scene,
                    **generated.model_dump(),
                ),
                analysis.scene,
            )
        return {
            "final_response": result.model_dump(),
            "messages": [
                {"role": "assistant", "content": result.recommended_reply}
            ],
        }
