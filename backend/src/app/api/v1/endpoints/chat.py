from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service, get_device_id
from app.schemas.chat import (
    ChatRequest,
    DeleteThreadResponse,
    ThreadDetail,
    ThreadSummary,
)
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])

OwnerId = Annotated[str, Depends(get_device_id)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@router.post("/chat")
def chat(
    request: ChatRequest,
    owner_id: OwnerId,
    service: ChatServiceDep,
) -> StreamingResponse:
    events = service.stream_chat(
        thread_id=request.thread_id,
        owner_id=owner_id,
        message=request.message,
    )
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/threads", response_model=list[ThreadSummary])
def list_threads(
    owner_id: OwnerId,
    service: ChatServiceDep,
) -> list[dict[str, str]]:
    return service.list_threads(owner_id)


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
def get_thread(
    thread_id: str,
    owner_id: OwnerId,
    service: ChatServiceDep,
) -> dict:
    return service.get_thread(thread_id, owner_id)


@router.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
def delete_thread(
    thread_id: str,
    owner_id: OwnerId,
    service: ChatServiceDep,
) -> dict[str, bool]:
    return service.delete_thread(thread_id, owner_id)
