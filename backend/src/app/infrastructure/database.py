from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ThreadStore:
    """Store business thread metadata independently from graph checkpoints."""

    def __init__(self, database_url: str, data_dir: Path):
        self._postgres = database_url.startswith(("postgresql://", "postgres://"))
        self._lock = threading.Lock()
        if self._postgres:
            import psycopg

            self._connection = psycopg.connect(database_url)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                data_dir / "threads.sqlite",
                check_same_thread=False,
            )
        self._execute(
            "CREATE TABLE IF NOT EXISTS threads ("
            "thread_id TEXT PRIMARY KEY, "
            "owner_id TEXT NOT NULL, "
            "title TEXT NOT NULL, "
            "created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS threads_owner_updated "
            "ON threads(owner_id, updated_at DESC)"
        )

    @property
    def marker(self) -> str:
        return "%s" if self._postgres else "?"

    def _execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(query, params)
            self._connection.commit()
            return cursor

    def claim(self, thread_id: str, owner_id: str, title: str) -> bool:
        now = datetime.now(UTC).isoformat()
        insert = (
            "INSERT INTO threads "
            "(thread_id, owner_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (thread_id) DO NOTHING"
            if self._postgres
            else "INSERT OR IGNORE INTO threads "
            "(thread_id, owner_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(insert, (thread_id, owner_id, title[:30], now, now))
            cursor.execute(
                f"SELECT owner_id FROM threads WHERE thread_id = {self.marker}",
                (thread_id,),
            )
            row = cursor.fetchone()
            if not row or row[0] != owner_id:
                self._connection.commit()
                return False
            cursor.execute(
                f"UPDATE threads SET updated_at = {self.marker} "
                f"WHERE thread_id = {self.marker}",
                (now, thread_id),
            )
            self._connection.commit()
            return True

    def owns(self, thread_id: str, owner_id: str) -> bool:
        row = self._execute(
            f"SELECT 1 FROM threads WHERE thread_id = {self.marker} "
            f"AND owner_id = {self.marker}",
            (thread_id, owner_id),
        ).fetchone()
        return bool(row)

    def list_for_owner(self, owner_id: str) -> list[tuple[str, str]]:
        return self._execute(
            f"SELECT thread_id, title FROM threads "
            f"WHERE owner_id = {self.marker} ORDER BY updated_at DESC",
            (owner_id,),
        ).fetchall()

    def delete(self, thread_id: str, owner_id: str) -> bool:
        cursor = self._execute(
            f"DELETE FROM threads WHERE thread_id = {self.marker} "
            f"AND owner_id = {self.marker}",
            (thread_id, owner_id),
        )
        return cursor.rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._connection.close()
