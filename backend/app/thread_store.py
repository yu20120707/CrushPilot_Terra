import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


class ThreadStore:
    """Business thread metadata; LangGraph checkpoints remain workflow-only."""

    def __init__(self, database_url: str, data_dir: Path):
        self.postgres = database_url.startswith("postgresql")
        self.lock = threading.Lock()
        if self.postgres:
            import psycopg
            self.connection = psycopg.connect(database_url)
        else:
            self.connection = sqlite3.connect(data_dir / "threads.sqlite", check_same_thread=False)
        self._execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        self._execute("CREATE INDEX IF NOT EXISTS threads_owner_updated ON threads(owner_id, updated_at DESC)")

    def _execute(self, query: str, params: tuple = ()):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            self.connection.commit()
            return cursor

    def claim(self, thread_id: str, owner_id: str, title: str) -> bool:
        now = datetime.now(UTC).isoformat()
        marker = "%s" if self.postgres else "?"
        insert = (
            "INSERT INTO threads (thread_id, owner_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (thread_id) DO NOTHING"
            if self.postgres
            else "INSERT OR IGNORE INTO threads (thread_id, owner_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)"
        )
        # Keep insert/read/update in one transaction. Locking each statement separately
        # allowed competing callers to observe a partially completed ownership claim.
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute(insert, (thread_id, owner_id, title[:30], now, now))
            row = cursor.execute(f"SELECT owner_id FROM threads WHERE thread_id = {marker}", (thread_id,)).fetchone()
            if not row or row[0] != owner_id:
                self.connection.commit()
                return False
            cursor.execute(f"UPDATE threads SET updated_at = {marker} WHERE thread_id = {marker}", (now, thread_id))
            self.connection.commit()
            return True

    def owns(self, thread_id: str, owner_id: str) -> bool:
        marker = "%s" if self.postgres else "?"
        row = self._execute(f"SELECT 1 FROM threads WHERE thread_id = {marker} AND owner_id = {marker}", (thread_id, owner_id)).fetchone()
        return bool(row)

    def list_for_owner(self, owner_id: str) -> list[tuple[str, str]]:
        marker = "%s" if self.postgres else "?"
        return self._execute(f"SELECT thread_id, title FROM threads WHERE owner_id = {marker} ORDER BY updated_at DESC", (owner_id,)).fetchall()

    def delete(self, thread_id: str, owner_id: str) -> bool:
        marker = "%s" if self.postgres else "?"
        cursor = self._execute(f"DELETE FROM threads WHERE thread_id = {marker} AND owner_id = {marker}", (thread_id, owner_id))
        return cursor.rowcount == 1
