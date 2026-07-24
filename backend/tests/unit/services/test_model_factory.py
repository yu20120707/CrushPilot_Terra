from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest

from app.agents.assistant.schemas import IntentAnalysis
from app.core.config import get_settings
from app.infrastructure.model_factory import JsonModelClient


def configured_client() -> JsonModelClient:
    settings = replace(
        get_settings(),
        model_base_url="https://model.example",
        model_api_key="key",
        model_name="model",
    )
    return JsonModelClient(settings)


def test_model_retries_transport_error_then_succeeds(monkeypatch) -> None:
    good = Mock()
    good.raise_for_status.return_value = None
    good.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"scene":"回应","goal":"回复","features":["聊天"],'
                        '"keywords":["沟通"],"risk_flags":[]}'
                    )
                }
            }
        ]
    }
    post = Mock(side_effect=[httpx.ConnectError("offline"), good])
    monkeypatch.setattr("app.infrastructure.model_factory.httpx.post", post)
    monkeypatch.setattr("app.infrastructure.model_factory.time.sleep", Mock())

    assert configured_client().call_json(
        "system", "user", IntentAnalysis
    ).scene == "回应"
    assert post.call_count == 2


def test_model_does_not_retry_client_error(monkeypatch) -> None:
    request = httpx.Request("POST", "https://model.example/chat/completions")
    response = httpx.Response(400, request=request)
    bad = Mock()
    bad.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad",
        request=request,
        response=response,
    )
    post = Mock(return_value=bad)
    monkeypatch.setattr("app.infrastructure.model_factory.httpx.post", post)

    with pytest.raises(RuntimeError, match="模型响应不可用"):
        configured_client().call_json("system", "user", IntentAnalysis)
    assert post.call_count == 1
