import threading
import time
import unittest

from app.services.chat_service import (
    ChatCommand,
    ChatService,
    ThreadExecutionLocks,
    _LeasedEvents,
)


class _Store:
    def __init__(self):
        self.deleted = False
        self.cancelled = False

    def owns(self, *_):
        return True

    def delete(self, *_):
        self.deleted = True
        return True

    def begin_delete(self, *_):
        return True

    def cancel_delete(self, *_):
        self.cancelled = True

    def finalize_delete(self, *_):
        self.deleted = True
        return True

    def claim(self, *_):
        return True


class _Checkpoint:
    def delete_thread(self, _):
        raise RuntimeError("checkpoint unavailable")


class ChatServiceTests(unittest.TestCase):
    def test_same_thread_lock_waits_and_reclaims_idle_entry(self):
        locks = ThreadExecutionLocks()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def first():
            with locks.hold("thread-1"):
                first_entered.set()
                release_first.wait(1)

        def second():
            with locks.hold("thread-1"):
                second_entered.set()

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        self.assertTrue(first_entered.wait(1))
        second_thread.start()
        time.sleep(0.05)
        self.assertFalse(second_entered.is_set())
        release_first.set()
        first_thread.join(1)
        second_thread.join(1)
        self.assertTrue(second_entered.is_set())
        self.assertFalse(locks._entries)

    def test_checkpoint_failure_keeps_thread_metadata(self):
        store = _Store()
        service = ChatService(
            graph=object(),
            graph_lock=threading.RLock(),
            checkpointer=_Checkpoint(),
            thread_store=store,
            retrieval_service=object(),
            local_sqlite_mode=False,
            readiness_errors=[],
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
            service.delete_thread("thread-1", "owner-1")
        self.assertFalse(store.deleted)
        self.assertTrue(store.cancelled)

    def test_unstarted_stream_can_be_closed_without_leaking_its_lock(self):
        locks = ThreadExecutionLocks()
        service = ChatService(
            graph=object(),
            graph_lock=threading.RLock(),
            checkpointer=object(),
            thread_store=_Store(),
            retrieval_service=object(),
            local_sqlite_mode=False,
            readiness_errors=[],
            thread_locks=locks,
        )
        stream = service.stream_chat(
            ChatCommand("thread-1", "hello", None, None, [], []),
            "owner-1",
        )
        stream.close()
        self.assertFalse(locks._entries)

    def test_close_during_active_generator_still_releases_lease(self):
        entered = threading.Event()
        unblock = threading.Event()

        def events():
            yield "start"
            entered.set()
            unblock.wait(1)
            yield "complete"

        locks = ThreadExecutionLocks()
        stream = _LeasedEvents(events(), locks.acquire("thread-1"))
        self.assertEqual(next(stream), "start")
        worker = threading.Thread(target=lambda: next(stream))
        worker.start()
        self.assertTrue(entered.wait(1))
        stream.close()
        self.assertFalse(locks._entries)
        unblock.set()
        worker.join(1)
