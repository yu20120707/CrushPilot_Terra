from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app.knowledge.retrieval.local_json import LocalJsonRetrievalService


class RuntimeNotReadyError(RuntimeError):
    pass


class ThreadNotFoundError(LookupError):
    pass


class LocalTraceUnavailableError(LookupError):
    pass


class LocalTraceMissingError(LookupError):
    pass


@dataclass(frozen=True)
class ChatCommand:
    thread_id: str
    message: str
    relationship_id: str | None
    conversation_summary: str | None
    relationship_facts: list[dict[str, Any]]
    user_preferences: list[dict[str, Any]]


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class _ThreadLease:
    def __init__(self, locks: "ThreadExecutionLocks", thread_id: str, entry: _LockEntry):
        self._locks = locks
        self._thread_id = thread_id
        self._entry = entry
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._entry.lock.release()
        self._locks._release(self._thread_id, self._entry)


class ThreadExecutionLocks:
    """Serializes one conversation without retaining idle lock objects."""

    def __init__(self):
        self._guard = threading.Lock()
        self._entries: dict[str, _LockEntry] = {}

    def acquire(self, thread_id: str) -> _ThreadLease:
        with self._guard:
            entry = self._entries.setdefault(thread_id, _LockEntry())
            # 在等待线程锁前计数，避免先到请求释放时移除仍被等待请求使用的锁对象。
            entry.users += 1
        entry.lock.acquire()
        return _ThreadLease(self, thread_id, entry)

    @contextmanager
    def hold(self, thread_id: str) -> Iterator[None]:
        lease = self.acquire(thread_id)
        try:
            yield
        finally:
            lease.release()

    def _release(self, thread_id: str, entry: _LockEntry) -> None:
        with self._guard:
            entry.users -= 1
            if entry.users == 0 and self._entries.get(thread_id) is entry:
                self._entries.pop(thread_id)


class _LeasedEvents(Iterator[str]):
    def __init__(self, events: Iterator[str], lease: _ThreadLease):
        self._events = events
        self._lease = lease
        self._closed = False

    def __iter__(self) -> "_LeasedEvents":
        return self

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration
        try:
            return next(self._events)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._events.close()
        except ValueError:
            # 同步图执行器仍可能占有生成器；无论关闭是否成功都必须释放会话租约。
            pass
        finally:
            self._lease.release()


class ChatService:
    """Application boundary for chat, thread ownership, and graph state."""

    def __init__(
        self,
        *,
        graph: Any,
        graph_lock: Any,
        checkpointer: Any,
        thread_store: Any,
        retrieval_service: Any,
        local_sqlite_mode: bool,
        readiness_errors: list[str],
        thread_locks: ThreadExecutionLocks | None = None,
    ):
        self._graph = graph
        self._graph_lock = graph_lock
        self._checkpointer = checkpointer
        self._thread_store = thread_store
        self._retrieval_service = retrieval_service
        self._local_sqlite_mode = local_sqlite_mode
        self._readiness_errors = readiness_errors
        self._thread_locks = thread_locks or ThreadExecutionLocks()
        self._draining = False
        self._draining_lock = threading.Lock()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def stream_chat(self, command: ChatCommand, owner_id: str) -> Iterator[str]:
        if self._readiness_errors or self._is_draining():
            raise RuntimeNotReadyError()
        lease = self._thread_locks.acquire(command.thread_id)
        try:
            # 获取会话租约后再次检查状态并完成 ownership claim，保证同会话的流式写入、
            # 读取和删除严格串行。
            if self._readiness_errors or self._is_draining():
                raise RuntimeNotReadyError()
            if not self._thread_store.claim(command.thread_id, owner_id, command.message):
                raise ThreadNotFoundError(command.thread_id)
        except Exception:
            lease.release()
            raise

        def events() -> Iterator[str]:
            # SSE 顺序是客户端协议的一部分：无论图执行成败，都以 end 收尾。
            yield "event: start\ndata: {}\n\n"
            try:
                with self._graph_lock:
                    if self._is_draining() or not self._thread_store.owns(
                        command.thread_id, owner_id
                    ):
                        raise ThreadNotFoundError(command.thread_id)
                    result = self._graph.invoke(
                        {
                            "user_message": command.message,
                            "device_id": owner_id,
                            "conversation_id": command.thread_id,
                            "relationship_id": command.relationship_id,
                            "conversation_summary": command.conversation_summary,
                            "relationship_facts": command.relationship_facts,
                            "user_preferences": command.user_preferences,
                            "messages": [{"role": "user", "content": command.message}],
                        },
                        self._config(command.thread_id),
                    )
                payload = json.dumps(result["final_response"], ensure_ascii=False)
                yield f"event: complete\ndata: {payload}\n\n"
            except Exception:
                payload = json.dumps(
                    {"message": "服务暂时不可用，请稍后重试。"}, ensure_ascii=False
                )
                yield f"event: error\ndata: {payload}\n\n"
            yield "event: end\ndata: {}\n\n"
        return _LeasedEvents(events(), lease)

    def list_threads(self, owner_id: str) -> list[dict[str, str]]:
        with self._graph_lock:
            self._require_active()
            return [
                {"thread_id": thread_id, "title": title}
                for thread_id, title in self._thread_store.list_for_owner(owner_id)
            ]

    def get_thread(self, thread_id: str, owner_id: str) -> dict[str, Any]:
        with self._thread_locks.hold(thread_id):
            with self._graph_lock:
                self._require_active()
                if not self._thread_store.owns(thread_id, owner_id):
                    raise ThreadNotFoundError(thread_id)
                state = self._graph.get_state(self._config(thread_id))
        return {"thread_id": thread_id, "messages": state.values.get("messages", [])}

    def delete_thread(self, thread_id: str, owner_id: str) -> dict[str, bool]:
        with self._thread_locks.hold(thread_id):
            with self._graph_lock:
                self._require_active()
                if not self._thread_store.begin_delete(thread_id, owner_id):
                    raise ThreadNotFoundError(thread_id)
                try:
                    # 先 tombstone 再删 checkpoint：删除期间线程立即不可见；checkpoint
                    # 失败则撤销 tombstone，避免隐藏仍可用的会话。
                    self._checkpointer.delete_thread(thread_id)
                except Exception:
                    self._thread_store.cancel_delete(thread_id, owner_id)
                    raise
                try:
                    # 元数据最终删除失败时保留 tombstone，启动阶段会重试，避免空状态会话复活。
                    self._thread_store.finalize_delete(thread_id, owner_id)
                except Exception:
                    pass
        return {"deleted": True}

    def local_trace(self, thread_id: str, owner_id: str) -> dict[str, Any]:
        with self._thread_locks.hold(thread_id):
            with self._graph_lock:
                self._require_active()
                if not self._thread_store.owns(thread_id, owner_id):
                    raise ThreadNotFoundError(thread_id)
                if not self._local_sqlite_mode or not isinstance(
                    self._retrieval_service, LocalJsonRetrievalService
                ):
                    raise LocalTraceUnavailableError(thread_id)
                trace = self._retrieval_service.trace_for(thread_id)
        if trace is None:
            raise LocalTraceMissingError(thread_id)
        return trace

    def begin_draining(self) -> None:
        with self._draining_lock:
            self._draining = True

    def _is_draining(self) -> bool:
        with self._draining_lock:
            return self._draining

    def _require_active(self) -> None:
        if self._is_draining():
            raise RuntimeNotReadyError()
