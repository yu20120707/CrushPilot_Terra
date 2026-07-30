import tempfile
import unittest
from pathlib import Path

from app.thread_store import ThreadStore


class ThreadStoreTests(unittest.TestCase):
    def test_tombstone_hides_then_can_cancel_or_finalize_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ThreadStore("", Path(directory))
            self.assertTrue(store.claim("thread-1", "owner-1", "title"))
            self.assertTrue(store.begin_delete("thread-1", "owner-1"))
            self.assertFalse(store.owns("thread-1", "owner-1"))
            self.assertEqual(store.list_for_owner("owner-1"), [])
            self.assertEqual(store.pending_deletions(), [("thread-1", "owner-1")])
            store.cancel_delete("thread-1", "owner-1")
            self.assertTrue(store.owns("thread-1", "owner-1"))
            self.assertTrue(store.begin_delete("thread-1", "owner-1"))
            self.assertTrue(store.finalize_delete("thread-1", "owner-1"))
            self.assertEqual(store.pending_deletions(), [])
            store.close()
