from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from app.infrastructure.database import ThreadStore
from app.services.chat_service import ChatService, ThreadNotFoundError


def test_thread_store_claim_is_atomic_between_owners(tmp_path) -> None:
    store = ThreadStore("", tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda owner: store.claim("same-thread", owner, "title"),
                    ["owner-a", "owner-b"],
                )
            )
        assert sorted(results) == [False, True]
    finally:
        store.close()


def test_chat_service_owns_graph_invocation_and_thread_lifecycle(tmp_path) -> None:
    graph = Mock()
    graph.invoke.return_value = {
        "final_response": {
            "intent": "聊天回应",
            "recommended_reply": "辛苦了。",
        }
    }
    graph.get_state.return_value = SimpleNamespace(
        values={"messages": [{"role": "assistant", "content": "辛苦了。"}]}
    )
    checkpointer = Mock()
    store = ThreadStore("", tmp_path)
    service = ChatService(graph, checkpointer, store)
    try:
        events = "".join(
            service.stream_chat(
                thread_id="thread-1",
                owner_id="owner-1",
                message="她很累",
            )
        )
        assert "event: complete" in events
        assert graph.invoke.call_count == 1
        assert service.list_threads("owner-1")[0]["thread_id"] == "thread-1"
        assert service.get_thread("thread-1", "owner-1")["messages"]
        assert service.delete_thread("thread-1", "owner-1") == {"deleted": True}
        checkpointer.delete_thread.assert_called_once_with("thread-1")
    finally:
        store.close()


def test_chat_service_rejects_a_thread_owned_by_another_device(tmp_path) -> None:
    store = ThreadStore("", tmp_path)
    service = ChatService(Mock(), Mock(), store)
    try:
        list(
            service.stream_chat(
                thread_id="shared",
                owner_id="owner-a",
                message="hello",
            )
        )
        try:
            service.stream_chat(
                thread_id="shared",
                owner_id="owner-b",
                message="steal",
            )
        except ThreadNotFoundError:
            pass
        else:
            raise AssertionError("cross-owner access should fail")
    finally:
        store.close()
