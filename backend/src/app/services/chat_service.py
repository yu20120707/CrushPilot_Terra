from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from app.infrastructure.database import ThreadStore

logger = logging.getLogger(__name__)


class ThreadNotFoundError(LookupError):
    """The requested thread does not exist for the current owner."""


class ChatService:
    """Application boundary for every chat and thread operation."""

    def __init__(self, graph: Any, checkpointer: Any, thread_store: ThreadStore):
        self._graph = graph
        self._checkpointer = checkpointer
        self._thread_store = thread_store

    @staticmethod
    def _graph_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def stream_chat(
        self,
        *,
        thread_id: str,
        owner_id: str,
        message: str,
    ) -> Iterator[str]:
        if not self._thread_store.claim(thread_id, owner_id, message):
            raise ThreadNotFoundError(thread_id)

        def events() -> Iterator[str]:
            yield "event: start\ndata: {}\n\n"
            try:
                result = self._graph.invoke(
                    {
                        "user_message": message,
                        "device_id": owner_id,
                        "messages": [{"role": "user", "content": message}],
                    },
                    self._graph_config(thread_id),
                )
                payload = json.dumps(
                    result["final_response"],
                    ensure_ascii=False,
                )
                yield f"event: complete\ndata: {payload}\n\n"
            except Exception:
                logger.exception("Chat graph failed for thread %s", thread_id)
                payload = json.dumps(
                    {"message": "服务暂时不可用，请稍后重试。"},
                    ensure_ascii=False,
                )
                yield f"event: error\ndata: {payload}\n\n"
            yield "event: end\ndata: {}\n\n"

        return events()

    def list_threads(self, owner_id: str) -> list[dict[str, str]]:
        return [
            {"thread_id": thread_id, "title": title}
            for thread_id, title in self._thread_store.list_for_owner(owner_id)
        ]

    def get_thread(self, thread_id: str, owner_id: str) -> dict[str, Any]:
        if not self._thread_store.owns(thread_id, owner_id):
            raise ThreadNotFoundError(thread_id)
        state = self._graph.get_state(self._graph_config(thread_id))
        return {
            "thread_id": thread_id,
            "messages": state.values.get("messages", []),
        }

    def delete_thread(self, thread_id: str, owner_id: str) -> dict[str, bool]:
        if not self._thread_store.delete(thread_id, owner_id):
            raise ThreadNotFoundError(thread_id)
        self._checkpointer.delete_thread(thread_id)
        return {"deleted": True}
