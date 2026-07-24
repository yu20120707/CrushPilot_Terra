from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class JsonModelClient:
    """Small OpenAI-compatible client that returns validated JSON models."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self._settings.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": f"{system}\n请只输出一个 JSON 对象。",
                },
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }

    def call_json(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
    ) -> SchemaT:
        settings = self._settings
        if not (
            settings.model_base_url
            and settings.model_api_key
            and settings.model_name
        ):
            raise RuntimeError(
                "模型未配置：请设置 "
                f"{settings.model_provider} 对应的环境变量，"
                "或只在本地开启 DEMO_MODE=true。"
            )

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{settings.model_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.model_api_key}"},
                    json=self._payload(system, user),
                    timeout=httpx.Timeout(25, connect=5),
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return schema.model_validate_json(content)
            except httpx.HTTPStatusError as error:
                last_error = error
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


def create_model_client(settings: Settings) -> JsonModelClient:
    return JsonModelClient(settings)
