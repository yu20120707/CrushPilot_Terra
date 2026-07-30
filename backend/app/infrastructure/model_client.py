from __future__ import annotations

import json
import time
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelProbe(BaseModel):
    status: Literal["ok"]


class OpenAICompatibleJsonClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def payload(self, system: str, user: str) -> dict[str, object]:
        # 所有模型调用共用协议约束，健康探测也经过真实的鉴权、传输和 JSON 解析路径。
        return {
            "model": self._settings.model_name,
            "messages": [
                {"role": "system", "content": system + "\n请只输出一个 JSON 对象。"},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
        }

    def call_json(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        settings = self._settings
        if not (settings.model_base_url and settings.model_api_key and settings.model_name):
            raise RuntimeError("模型未配置")
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{settings.model_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.model_api_key}"},
                    json=self.payload(system, user),
                    timeout=httpx.Timeout(25, connect=5),
                    trust_env=settings.model_trust_env,
                )
                response.raise_for_status()
                return schema.model_validate_json(
                    response.json()["choices"][0]["message"]["content"]
                )
            except httpx.HTTPStatusError as error:
                last_error = error
                # 4xx（429 除外）通常是配置或请求错误，重试没有收益。
                if error.response.status_code < 500 and error.response.status_code != 429:
                    break
            except (
                httpx.TransportError,
                ValidationError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
            if attempt == 0:
                time.sleep(0.2)
        raise RuntimeError("模型响应不可用") from last_error

    def probe(self) -> None:
        # 不依赖各供应商不同的健康接口，而是验证最小的真实调用契约。
        self.call_json(
            "你是服务健康检查。",
            "返回 JSON：{\"status\": \"ok\"}。",
            ModelProbe,
        )
