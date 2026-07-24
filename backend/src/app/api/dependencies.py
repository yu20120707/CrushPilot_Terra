from typing import Annotated
from uuid import UUID

from fastapi import Header, HTTPException, Request

from app.services.chat_service import ChatService


def get_device_id(
    x_device_id: Annotated[str, Header(alias="X-Device-Id")],
) -> str:
    try:
        return str(UUID(x_device_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="无效设备标识") from error


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
