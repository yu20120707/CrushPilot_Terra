import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.chat_service import ThreadNotFoundError

logger = logging.getLogger(__name__)


async def thread_not_found_handler(
    _request: Request,
    _error: ThreadNotFoundError,
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "会话不存在"})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ThreadNotFoundError, thread_not_found_handler)
