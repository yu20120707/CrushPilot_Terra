from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class ThreadSummary(BaseModel):
    thread_id: str
    title: str


class ThreadDetail(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]


class DeleteThreadResponse(BaseModel):
    deleted: bool
