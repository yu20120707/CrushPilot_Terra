import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


class ThreadStore:
    """业务线程元数据；LangGraph checkpoint 仅保存工作流状态。

    两类数据存储独立，删除采用 tombstone 补偿，而不是假设存在跨存储事务。
    """

    def __init__(self, database_url: str, data_dir: Path):
        self.postgres = database_url.startswith("postgresql")
        self.lock = threading.Lock()
        if self.postgres:
            import psycopg
            self.connection = psycopg.connect(database_url)
        else:
            self.connection = sqlite3.connect(data_dir / "threads.sqlite", check_same_thread=False)
        try:
            self._execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleting INTEGER NOT NULL DEFAULT 0)")
            self._ensure_deleting_column()
            self._execute("CREATE INDEX IF NOT EXISTS threads_owner_updated ON threads(owner_id, updated_at DESC)")
        except Exception:
            self.connection.close()
            raise

    def _execute(self, query: str, params: tuple = ()):
        with self.lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, params)
                self.connection.commit()
                return cursor
            except Exception:
                self.connection.rollback()
                raise

    def claim(self, thread_id: str, owner_id: str, title: str) -> bool:
        now = datetime.now(UTC).isoformat()
        marker = "%s" if self.postgres else "?"
        insert = (
            "INSERT INTO threads (thread_id, owner_id, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (thread_id) DO NOTHING"
            if self.postgres
            else "INSERT OR IGNORE INTO threads (thread_id, owner_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)"
        )
        # 插入、归属校验和更新时间必须处于同一临界区；逐句加锁会让并发请求观察到
        # 尚未完成的 ownership claim。
        with self.lock:
            cursor = self.connection.cursor()
            try:
                cursor.execute(insert, (thread_id, owner_id, title[:30], now, now))
                row = cursor.execute(
                    f"SELECT owner_id FROM threads WHERE thread_id = {marker} AND deleting = 0",
                    (thread_id,),
                ).fetchone()
                if not row or row[0] != owner_id:
                    self.connection.commit()
                    return False
                cursor.execute(
                    f"UPDATE threads SET updated_at = {marker} WHERE thread_id = {marker}",
                    (now, thread_id),
                )
                self.connection.commit()
                return True
            except Exception:
                self.connection.rollback()
                raise

    def owns(self, thread_id: str, owner_id: str) -> bool:
        marker = "%s" if self.postgres else "?"
        row = self._execute(f"SELECT 1 FROM threads WHERE thread_id = {marker} AND owner_id = {marker} AND deleting = 0", (thread_id, owner_id)).fetchone()
        return bool(row)

    def list_for_owner(self, owner_id: str) -> list[tuple[str, str]]:
        marker = "%s" if self.postgres else "?"
        return self._execute(f"SELECT thread_id, title FROM threads WHERE owner_id = {marker} AND deleting = 0 ORDER BY updated_at DESC", (owner_id,)).fetchall()

    def delete(self, thread_id: str, owner_id: str) -> bool:
        marker = "%s" if self.postgres else "?"
        cursor = self._execute(f"DELETE FROM threads WHERE thread_id = {marker} AND owner_id = {marker}", (thread_id, owner_id))
        return cursor.rowcount == 1

    def begin_delete(self, thread_id: str, owner_id: str) -> bool:
        # tombstone 同时是删除中的可见性开关和失败恢复锚点。
        marker = "%s" if self.postgres else "?"
        cursor = self._execute(
            f"UPDATE threads SET deleting = 1 WHERE thread_id = {marker} AND owner_id = {marker} AND deleting = 0",
            (thread_id, owner_id),
        )
        return cursor.rowcount == 1

    def cancel_delete(self, thread_id: str, owner_id: str) -> None:
        marker = "%s" if self.postgres else "?"
        self._execute(
            f"UPDATE threads SET deleting = 0 WHERE thread_id = {marker} AND owner_id = {marker}",
            (thread_id, owner_id),
        )

    def finalize_delete(self, thread_id: str, owner_id: str) -> bool:
        marker = "%s" if self.postgres else "?"
        cursor = self._execute(
            f"DELETE FROM threads WHERE thread_id = {marker} AND owner_id = {marker} AND deleting = 1",
            (thread_id, owner_id),
        )
        return cursor.rowcount == 1

    def pending_deletions(self) -> list[tuple[str, str]]:
        return self._execute(
            "SELECT thread_id, owner_id FROM threads WHERE deleting = 1"
        ).fetchall()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def _ensure_deleting_column(self) -> None:
        # 兼容已存在的 SQLite/PostgreSQL 数据库；迁移保持幂等，避免要求人工重建。
        if self.postgres:
            self._execute(
                "ALTER TABLE threads ADD COLUMN IF NOT EXISTS deleting INTEGER NOT NULL DEFAULT 0"
            )
            return
        columns = self._execute("PRAGMA table_info(threads)").fetchall()
        if not any(column[1] == "deleting" for column in columns):
            self._execute(
                "ALTER TABLE threads ADD COLUMN deleting INTEGER NOT NULL DEFAULT 0"
            )
