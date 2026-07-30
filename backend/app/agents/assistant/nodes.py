from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from app.knowledge.domain.models import (
    CONTROLLED_TOPICS,
    ConfirmedFact,
    ConversationContext,
    ConversationMessage,
    EvidenceAssessment,
    RetrievalPlan,
    SceneAndRetrievalPlan,
    SceneSnapshot,
)
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.retrieval.context_assembler import assemble_prompt
from app.knowledge.retrieval.context_builder import build_context as make_context
from app.knowledge.skill.runtime import SkillRuntime

from .prompts import SAFETY_PROMPT, generation_system_prompt, scene_plan_prompt
from .schemas import ChatResult, ChatState


DANGEROUS_ACTION = (
    r"跟踪|尾随|蹲守|威胁|恐吓|强迫|逼迫|纠缠|骚扰|诈骗|冒充|灌醉|下药|偷拍"
)
INPUT_UNSAFE_PATTERN = re.compile(rf"{DANGEROUS_ACTION}|未成年")
MISSING_CONTEXT_PATTERN = re.compile(
    r"没有(?:前文|上下文)|(?:没有|没看到|未提供)原话|没有具体行为|"
    r"缺少(?:双方)?关系阶段|没有描述(?:具体)?冲突|"
    r"不知道(?:已经)?过去多久|不知道拒绝(?:了)?什么|"
    r"(?:没有|未)提供对方消息"
)
DANGEROUS_ADVICE_PATTERN = re.compile(DANGEROUS_ACTION)
INPUT_CESSATION_PATTERN = re.compile(
    rf"(?:停止|不再|不要|别再|避免|拒绝|放弃|摆脱)\s*"
    rf"(?:继续|再次|再)?\s*(?:去)?\s*"
    rf"(?:通过(?:小号|电话|短信|社交媒体|账号))?\s*"
    rf"(?:对(?:她|他|对方|前任)的)?\s*"
    rf"(?P<danger>{DANGEROUS_ACTION})"
)
CESSATION_PREFIX = re.compile(
    r"^\s*(?:(?:前任|对方)要求(?:我)?不再联系[，,]\s*)?"
    r"(?:我(?:需要|决定|想|会|必须|应该|要)|请|必须|应该|要)?\s*$"
)
CESSATION_SAFE_SUFFIX = re.compile(
    r"^\s*(?:(?:她|他|对方|前任)(?:回家)?(?:的(?:行为|冲动))?"
    r"|(?:这种|上述)?行为|的冲动)?\s*[。！？]?\s*$"
)
SCENE_POLICY_IDS = ("explicit_rejection", "emotional_support", "insufficient_information")
ALLOWED_HARD_FILTERS = {
    "review_status",
    "task_types",
    "relationship_stages",
    "knowledge_type",
}
MAX_OUTPUT_CHARACTERS = 20


def _shorten_output(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARACTERS:
        return text
    prefix = text[:MAX_OUTPUT_CHARACTERS]
    for ending in "。！？；":
        if (position := prefix.rfind(ending)) >= 4:
            return prefix[: position + 1]
    return prefix.rstrip("，、 ")


def _compact_result(result: ChatResult, intent: str) -> ChatResult:
    return result.model_copy(
        update={
            "intent": intent,
            "judgement": _shorten_output(result.judgement),
            "recommended_reply": _shorten_output(result.recommended_reply),
            "alternatives": [_shorten_output(item) for item in result.alternatives],
            "warning": _shorten_output(result.warning) if result.warning else None,
        }
    )


def safe_result(intent: str = "边界风险") -> ChatResult:
    return ChatResult(
        intent=intent,
        judgement="这事可能越界。",
        recommended_reply="先尊重对方，别再施压。",
        alternatives=["对方不想继续，就停。", "先冷静，别越界。"],
        primary_style="稳重",
        alternative_styles=["稳重", "稳重"],
        warning="不提供跟踪、骚扰等建议。",
    )


def contains_unsafe_advice(text: str) -> bool:
    return DANGEROUS_ADVICE_PATTERN.search(text) is not None


def contains_unsafe_input(text: str) -> bool:
    if "未成年" in text:
        return True
    dangerous = list(DANGEROUS_ADVICE_PATTERN.finditer(text))
    if len(dangerous) != 1:
        return bool(dangerous)
    danger = dangerous[0]
    for cessation in INPUT_CESSATION_PATTERN.finditer(text):
        if cessation.span("danger") != danger.span():
            continue
        prefix = CESSATION_PREFIX.fullmatch(text[: cessation.start()])
        if prefix is None:
            continue
        if CESSATION_SAFE_SUFFIX.fullmatch(text[danger.end() :]) is None:
            continue
        return False
    return True


def validate_result(result: ChatResult, intent: str) -> ChatResult:
    if contains_unsafe_advice(
        "\n".join(
            [
                result.judgement,
                result.recommended_reply,
                *result.alternatives,
                result.warning or "",
            ]
        )
    ):
        return _compact_result(safe_result(intent), intent)
    return _compact_result(result, intent)


def _normalize_fact(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).casefold()


def _enforce_known_fact_provenance(
    plan: SceneAndRetrievalPlan, context: ConversationContext
) -> None:
    sources = [
        context.current_message,
        *(message.content for message in context.recent_messages),
        context.conversation_summary or "",
        *(fact.content for fact in context.relationship_facts),
        *(fact.content for fact in context.user_preferences),
    ]
    normalized_sources = [
        normalized for source in sources if (normalized := _normalize_fact(source))
    ]
    traceable: list[str] = []
    unverified: list[str] = []
    for fact in plan.scene.known_facts:
        normalized = _normalize_fact(fact)
        target = (
            traceable
            if normalized and any(normalized in source for source in normalized_sources)
            else unverified
        )
        target.append(fact)
    plan.scene.known_facts = list(dict.fromkeys(traceable))
    plan.scene.key_unknowns = list(
        dict.fromkeys([*plan.scene.key_unknowns, *unverified])
    )


class AssistantNodes:
    def __init__(
        self,
        *,
        skill_runtime: SkillRuntime,
        retrieval_service: object,
        call_json: Callable[[str, str, type], Any],
        metrics: RetrievalMetrics,
        corpus_version: str,
        embedding_model: str | None,
        trace_writer: object | None = None,
        demo_mode: bool = False,
    ):
        self.skill_runtime = skill_runtime
        self.retrieval_service = retrieval_service
        self.call_json = call_json
        self.metrics = metrics
        self.corpus_version = corpus_version
        self.embedding_model = embedding_model
        self.trace_writer = trace_writer
        self.demo_mode = demo_mode

    def build_context(self, state: ChatState) -> ChatState:
        # 只带最近窗口，防止历史无限增长挤占检索证据和最终回答的 token 预算；
        # 更久的信息应由上层摘要或已确认事实承载。
        messages = [
            ConversationMessage.model_validate(message)
            for message in state.get("messages", [])[-12:]
        ]
        context = make_context(
            state["user_message"],
            messages,
            state.get("conversation_summary"),
            state.get("relationship_id"),
            [ConfirmedFact.model_validate(item) for item in state.get("relationship_facts", [])],
            [ConfirmedFact.model_validate(item) for item in state.get("user_preferences", [])],
        )
        return {"conversation_context": context.model_dump(mode="json")}

    def analyze_scene_and_plan(self, state: ChatState) -> ChatState:
        started = perf_counter()
        context = ConversationContext.model_validate(state["conversation_context"])
        unsafe_input = contains_unsafe_input(context.current_message)
        if unsafe_input:
            # 高风险输入不交给模型分类，使用确定性计划避免生成链路放大越界意图。
            plan = _unsafe_plan(context.current_message)
        elif self.demo_mode:
            plan = _demo_plan(context.current_message)
        else:
            # call_json owns the single schema/transport retry.
            skill = self.skill_runtime.view(SCENE_POLICY_IDS)
            try:
                plan = self.call_json(
                    SAFETY_PROMPT,
                    scene_plan_prompt(context, skill),
                    SceneAndRetrievalPlan,
                )
            except Exception:
                self._write_failure_trace(
                    state, "scene_analysis_failed", (perf_counter() - started) * 1000
                )
                raise
            finally:
                self.metrics.observe(
                    "scene_analysis_latency_ms", (perf_counter() - started) * 1000
                )
            _enforce_known_fact_provenance(plan, context)
            # 模型可以建议场景，但 skill 的安全策略拥有最终约束权，不能被模型输出绕过。
            active = [
                scene_id
                for scene_id in plan.scene.active_skill_scenarios
                if scene_id in skill.scene_policies
            ]
            if (
                "explicit_rejection" in active
                and plan.scene.task_type != "relationship_exit"
            ):
                plan.scene.task_type = "boundary"
                plan.scene.recommended_action = "stop"
            elif (
                "insufficient_information" in active
                or (
                    not active
                    and MISSING_CONTEXT_PATTERN.search(context.current_message)
                )
            ):
                active = ["insufficient_information"]
                plan.scene.task_type = "general_advice"
                plan.scene.recommended_action = "clarify"
                plan.retrieval.required_topics = []
                plan.retrieval.excluded_topics = []
            required = list(plan.retrieval.required_topics)
            excluded = list(plan.retrieval.excluded_topics)
            required = [topic for topic in required if topic in CONTROLLED_TOPICS]
            excluded = [topic for topic in excluded if topic in CONTROLLED_TOPICS]
            excluded = list(dict.fromkeys(excluded))
            model_excluded = set(excluded)
            required = [
                topic for topic in dict.fromkeys(required) if topic not in model_excluded
            ]
            policy_required: list[str] = []
            policy_excluded: list[str] = []
            for scene_id in active:
                policy = skill.scene_policies[scene_id]
                policy_required.extend(policy.get("required_topics", []))
                policy_excluded.extend(policy.get("excluded_topics", []))
            policy_required = list(dict.fromkeys(policy_required))
            policy_excluded = list(dict.fromkeys(policy_excluded))
            policy_overlap = set(policy_required) & set(policy_excluded)
            if policy_overlap:
                raise ValueError(
                    f"skill policy topics cannot be both required and excluded: "
                    f"{sorted(policy_overlap)}"
                )
            required = [
                topic for topic in required if topic not in set(policy_excluded)
            ]
            excluded = [
                topic for topic in excluded if topic not in set(policy_required)
            ]
            required.extend(policy_required)
            excluded.extend(policy_excluded)
            plan.scene.active_skill_scenarios = active
            plan.retrieval.required_topics = list(dict.fromkeys(required))
            plan.retrieval.excluded_topics = list(dict.fromkeys(excluded))
            plan.retrieval.hard_filters = {
                key: value
                for key, value in plan.retrieval.hard_filters.items()
                if key in ALLOWED_HARD_FILTERS
                and isinstance(value, list)
                and value
                and all(isinstance(item, str) and item for item in value)
            }
            plan.retrieval.hard_filters["review_status"] = ["approved"]
        if self.demo_mode or unsafe_input:
            self.metrics.observe(
                "scene_analysis_latency_ms", (perf_counter() - started) * 1000
            )
        return {"scene_and_retrieval_plan": plan.model_dump()}

    def retrieve_evidence(self, state: ChatState) -> ChatState:
        plan = SceneAndRetrievalPlan.model_validate(state["scene_and_retrieval_plan"])
        skill = self.skill_runtime.view(plan.scene.active_skill_scenarios)
        started = perf_counter()
        try:
            # 检索 trace 与模型生成分别记录失败，便于区分“无证据”和“模型不可用”。
            result = self.retrieval_service.retrieve(
                request_id=str(uuid4()),
                conversation_id=state.get("conversation_id", "unknown"),
                skill_version=skill.version,
                skill_sha256=skill.source_sha256,
                corpus_version=self.corpus_version,
                scene=plan.scene,
                plan=plan.retrieval,
                embedding_model=self.embedding_model,
            )
        except Exception:
            elapsed = (perf_counter() - started) * 1000
            self.metrics.observe("retrieval_failure_latency_ms", elapsed)
            self._write_failure_trace(state, "retrieval_failed", elapsed)
            raise
        return {
            "evidence_chunks": result.chunks,
            "evidence_assessment": result.assessment.model_dump(),
            "retrieval_trace_id": result.trace.request_id,
        }

    def generate_answer(self, state: ChatState) -> ChatState:
        plan = SceneAndRetrievalPlan.model_validate(state["scene_and_retrieval_plan"])
        if contains_unsafe_input(state["user_message"]):
            result = safe_result(plan.scene.task_type)
        elif self.demo_mode:
            result = ChatResult(
                intent=plan.scene.task_type,
                judgement="狗头军师建议先接住当下情绪，再给对方留出空间。",
                recommended_reply="听起来你今天挺累的，先好好休息，等你有空再聊。",
                alternatives=["辛苦啦，先让自己放松一下。", "不用急着回复，忙完再说。"],
            )
        else:
            context = ConversationContext.model_validate(state["conversation_context"])
            assessment = EvidenceAssessment.model_validate(state["evidence_assessment"])
            skill = self.skill_runtime.view(plan.scene.active_skill_scenarios)
            prompt = assemble_prompt(
                # 这里只接收 evidence gate 放行的 chunks，不能把原始召回候选当作事实来源。
                list(skill.core_rules),
                skill.scene_policies,
                skill.output_policy,
                plan.scene,
                assessment,
                state.get("evidence_chunks", []),
                context,
                ChatResult.model_json_schema(),
            )
            started = perf_counter()
            try:
                result = self.call_json(
                    generation_system_prompt(skill), prompt, ChatResult
                )
            except Exception:
                self._mark_trace_failure(
                    state.get("retrieval_trace_id"),
                    "generation_failed",
                    (perf_counter() - started) * 1000,
                )
                raise
            finally:
                self.metrics.observe(
                    "generation_latency_ms", (perf_counter() - started) * 1000
                )
        return {"final_response": result.model_dump()}

    def validate_output(self, state: ChatState) -> ChatState:
        # 最终防线独立于模型提示：即使 schema 校验通过，也必须做安全替换和长度收敛。
        plan = SceneAndRetrievalPlan.model_validate(state["scene_and_retrieval_plan"])
        result = validate_result(
            ChatResult.model_validate(state["final_response"]), plan.scene.task_type
        )
        return {
            "final_response": result.model_dump(),
            "messages": [{"role": "assistant", "content": result.recommended_reply}],
        }

    def _write_failure_trace(
        self, state: ChatState, reason: str, latency_ms: float
    ) -> None:
        if self.trace_writer is None:
            return
        try:
            skill = self.skill_runtime.view(())
            latency_key = {
                "scene_analysis_failed": "scene_analysis_latency_ms",
                "retrieval_failed": "retrieval_failure_latency_ms",
                "generation_failed": "generation_latency_ms",
            }[reason]
            trace = {
                "request_id": str(uuid4()),
                "conversation_id": state.get("conversation_id", "unknown"),
                "skill_version": skill.version,
                "skill_sha256": skill.source_sha256,
                "corpus_version": self.corpus_version,
                "scene_snapshot": {},
                "retrieval_plan": {},
                "lexical_candidates": [],
                "vector_candidates": [],
                "fused_candidates": [],
                "reranked_candidates": [],
                "selected_chunks": [],
                "evidence_assessment": {"status": "insufficient"},
                "fallback_reason": reason,
                "latency_ms": {latency_key: latency_ms},
            }
            self.trace_writer.write_trace(trace)
        except Exception:
            self.metrics.observe("trace_write_failure_count", 1)

    def _mark_trace_failure(
        self, trace_id: str | None, reason: str, latency_ms: float
    ) -> None:
        if not trace_id or self.trace_writer is None:
            return
        try:
            self.trace_writer.mark_trace_failure(trace_id, reason, latency_ms)
        except Exception:
            self.metrics.observe("trace_write_failure_count", 1)


def _base_retrieval(query: str, required: list[str] | None = None) -> RetrievalPlan:
    return RetrievalPlan(
        queries=[query],
        required_topics=required or [],
        optional_topics=[],
        excluded_topics=["manipulation", "aggressive_pursuit"],
        hard_filters={"review_status": ["approved"]},
        soft_preferences={},
    )


def _demo_plan(message: str) -> SceneAndRetrievalPlan:
    return SceneAndRetrievalPlan(
        scene=SceneSnapshot(
            task_type="reply",
            recommended_action="respond",
            current_event=message,
            user_goal="给出自然、尊重边界的回应",
            known_facts=[message],
            uncertain_inferences=[],
            key_unknowns=[],
            counterpart_signals=[],
            explicit_boundaries=[],
            active_skill_scenarios=[],
            confidence=1,
        ),
        retrieval=_base_retrieval(f"{message} 如何低压力回应"),
    )


def _unsafe_plan(message: str) -> SceneAndRetrievalPlan:
    plan = _demo_plan(message)
    plan.scene.task_type = "boundary"
    plan.scene.recommended_action = "stop"
    plan.scene.user_goal = "拒绝越界请求并提供安全替代"
    plan.scene.explicit_boundaries = ["不得伤害、欺骗、跟踪或强迫他人"]
    plan.retrieval = _base_retrieval("关系边界与安全替代", ["boundary"])
    return plan
