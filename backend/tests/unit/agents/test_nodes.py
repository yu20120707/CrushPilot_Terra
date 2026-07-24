from dataclasses import replace
from unittest.mock import Mock

from app.agents.assistant import tools
from app.agents.assistant.edges import route_after_retrieval
from app.agents.assistant.graph import build_graph
from app.agents.assistant.nodes import (
    contains_unsafe_advice,
    safe_result,
    validate_result,
)
from app.agents.assistant.prompts import CHAT_SYSTEM_PROMPT
from app.agents.assistant.schemas import (
    ChatResult,
    GeneratedChatResult,
    IntentAnalysis,
)
from app.core.config import get_settings


def test_retrieval_edge_refines_once() -> None:
    assert route_after_retrieval(
        {"retrieved_knowledge": [], "retrieval_attempt": 0}
    ) == "refine"
    assert route_after_retrieval(
        {"retrieved_knowledge": [], "retrieval_attempt": 1}
    ) == "generate"


def test_safe_negation_is_allowed_but_executable_advice_is_rejected() -> None:
    result = ChatResult(
        intent="ignored",
        judgement="正常",
        recommended_reply="你可以威胁她。",
        alternatives=["继续施压。", "别放弃。"],
    )
    assert validate_result(result, "聊天回应").warning is not None
    assert not contains_unsafe_advice("不要强迫或骚扰对方。")
    for advice in [
        "每天在她公司楼下等她",
        "别停，反复给她发消息直到回应",
        "灌醉后再表白",
        "假装偶遇来接近她",
        "偷拍视频留作把柄",
    ]:
        assert contains_unsafe_advice(advice), advice


def test_safety_shortcut_keeps_the_assistant_identity() -> None:
    result = safe_result()
    assert result.skill == "goutoujunshi"
    assert "狗头军师判断" in result.judgement


def test_generated_result_ignores_an_echoed_intent_object() -> None:
    generated = GeneratedChatResult.model_validate_json(
        """{
            "intent": {"scene": "恋爱聊天", "risk_flags": []},
            "judgement": "先回应对方的状态。",
            "recommended_reply": "辛苦了，忙完再聊。",
            "alternatives": ["先好好休息。", "不用急着回复。"]
        }"""
    )
    assert generated.recommended_reply == "辛苦了，忙完再聊。"


def test_production_graph_calls_model_for_intent_and_response(monkeypatch) -> None:
    settings = replace(get_settings(), demo_mode=False)
    analysis = IntentAnalysis(
        scene="回应疲惫",
        goal="低压力回应",
        features=["疲惫"],
        keywords=["累"],
        risk_flags=["unknown_flag"],
    )
    reply = GeneratedChatResult(
        judgement="共情。",
        recommended_reply="辛苦了，早点休息。",
        alternatives=["先放松一下。", "忙完再聊。"],
    )
    model = Mock()
    model.call_json.side_effect = [analysis, reply]
    evidence = [
        {
            "source": "card:low-pressure-chat",
            "title": "低压力聊天",
            "content": "尊重边界",
            "tier": "safe",
        }
    ]
    monkeypatch.setattr(tools, "retrieve_by_keywords", Mock(return_value=evidence))
    monkeypatch.setattr(tools, "retrieve_by_vector", Mock(return_value=[]))

    graph = build_graph(settings=settings, model_client=model)
    result = graph.invoke(
        {
            "user_message": "她很累",
            "device_id": "owner",
            "messages": [{"role": "user", "content": "她很累"}],
        }
    )

    assert model.call_json.call_count == 2
    assert model.call_json.call_args_list[1].args[0] == CHAT_SYSTEM_PROMPT
    assert "<GOUTOUJUNSHI_REFERENCES>" in model.call_json.call_args_list[1].args[1]
    assert result["final_response"]["recommended_reply"] == "辛苦了，早点休息。"
