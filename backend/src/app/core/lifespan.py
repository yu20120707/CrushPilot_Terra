from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.assistant.graph import build_graph
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.infrastructure.checkpointer import checkpointer_context
from app.infrastructure.database import ThreadStore
from app.infrastructure.model_factory import create_model_client
from app.services.chat_service import ChatService


def create_lifespan(
    settings: Settings | None = None,
) -> Callable[[FastAPI], AsyncIterator[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or get_settings()
        configure_logging(active_settings)
        active_settings.data_dir.mkdir(parents=True, exist_ok=True)
        with checkpointer_context(active_settings) as checkpointer:
            thread_store = ThreadStore(
                active_settings.database_url,
                active_settings.data_dir,
            )
            graph = build_graph(
                settings=active_settings,
                model_client=create_model_client(active_settings),
                checkpointer=checkpointer,
            )
            app.state.settings = active_settings
            app.state.chat_service = ChatService(
                graph=graph,
                checkpointer=checkpointer,
                thread_store=thread_store,
            )
            try:
                yield
            finally:
                thread_store.close()

    return lifespan
