from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from app.api.v1.dependencies import device_id, runtime
from app.bootstrap.runtime import AppRuntime
from app.services.chat_service import (
    ChatCommand,
    LocalTraceMissingError,
    LocalTraceUnavailableError,
    RuntimeNotReadyError,
    ThreadNotFoundError,
)


router = APIRouter()


class ClosingStreamingResponse(StreamingResponse):
    """在客户端断开或框架取消响应时关闭同步事件迭代器。"""
    def __init__(self, events, **kwargs):
        super().__init__(events, **kwargs)
        self._events = events

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # StreamingResponse 不保证完整消费同步迭代器；显式 close 才能归还会话租约。
            self._events.close()


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    relationship_id: str | None = Field(default=None, max_length=100)
    conversation_summary: str | None = Field(default=None, max_length=4000)
    relationship_facts: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    user_preferences: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(response: Response, app_runtime: AppRuntime = Depends(runtime)) -> dict[str, object]:
    # /health 仅说明进程存活；/ready 才代表模型、检索和持久化依赖可接收流量。
    payload = app_runtime.readiness()
    if not payload["ready"]:
        response.status_code = 503
    return payload


@router.get("/metrics")
def metrics(app_runtime: AppRuntime = Depends(runtime)) -> dict[str, dict[str, float]]:
    return app_runtime.metrics.snapshot()


@router.post("/api/v1/chat")
def chat(
    request: ChatRequest,
    owner_id: str = Depends(device_id),
    app_runtime: AppRuntime = Depends(runtime),
) -> StreamingResponse:
    command = ChatCommand(
        thread_id=request.thread_id,
        message=request.message,
        relationship_id=request.relationship_id,
        conversation_summary=request.conversation_summary,
        relationship_facts=request.relationship_facts,
        user_preferences=request.user_preferences,
    )
    try:
        events = app_runtime.chat_service.stream_chat(command, owner_id)
    except RuntimeNotReadyError as error:
        raise HTTPException(status_code=503, detail="服务尚未就绪") from error
    except ThreadNotFoundError as error:
        raise HTTPException(status_code=404, detail="会话不存在") from error
    return ClosingStreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/v1/threads")
def list_threads(
    owner_id: str = Depends(device_id), app_runtime: AppRuntime = Depends(runtime)
) -> list[dict[str, str]]:
    try:
        return app_runtime.chat_service.list_threads(owner_id)
    except RuntimeNotReadyError as error:
        raise HTTPException(status_code=503, detail="服务正在停止") from error


@router.get("/api/v1/local-traces/{thread_id}")
def local_trace(
    thread_id: str,
    owner_id: str = Depends(device_id),
    app_runtime: AppRuntime = Depends(runtime),
) -> dict[str, Any]:
    try:
        return app_runtime.chat_service.local_trace(thread_id, owner_id)
    except RuntimeNotReadyError as error:
        raise HTTPException(status_code=503, detail="服务正在停止") from error
    except ThreadNotFoundError as error:
        raise HTTPException(status_code=404, detail="会话不存在") from error
    except LocalTraceUnavailableError as error:
        raise HTTPException(status_code=404, detail="本地检索追踪不可用") from error
    except LocalTraceMissingError as error:
        raise HTTPException(status_code=404, detail="尚无检索追踪") from error


@router.get("/api/v1/threads/{thread_id}")
def get_thread(
    thread_id: str,
    owner_id: str = Depends(device_id),
    app_runtime: AppRuntime = Depends(runtime),
) -> dict[str, Any]:
    try:
        return app_runtime.chat_service.get_thread(thread_id, owner_id)
    except RuntimeNotReadyError as error:
        raise HTTPException(status_code=503, detail="服务正在停止") from error
    except ThreadNotFoundError as error:
        raise HTTPException(status_code=404, detail="会话不存在") from error


@router.delete("/api/v1/threads/{thread_id}")
def delete_thread(
    thread_id: str,
    owner_id: str = Depends(device_id),
    app_runtime: AppRuntime = Depends(runtime),
) -> dict[str, bool]:
    try:
        return app_runtime.chat_service.delete_thread(thread_id, owner_id)
    except RuntimeNotReadyError as error:
        raise HTTPException(status_code=503, detail="服务正在停止") from error
    except ThreadNotFoundError as error:
        raise HTTPException(status_code=404, detail="会话不存在") from error
