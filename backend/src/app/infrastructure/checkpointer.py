from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from app.core.config import Settings


@contextmanager
def checkpointer_context(settings: Settings) -> Iterator[Any]:
    """Create and close the environment-appropriate LangGraph checkpointer."""

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith(("postgresql://", "postgres://")):
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            checkpointer.setup()
            yield checkpointer
        return

    connection = sqlite3.connect(
        settings.data_dir / "checkpoints.sqlite",
        check_same_thread=False,
    )
    try:
        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        yield checkpointer
    finally:
        connection.close()
