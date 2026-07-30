from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.assistant.nodes import contains_unsafe_advice, safe_result, validate_result
from app.agents.assistant.schemas import ChatResult
from app.api.v1.routes import router
from app.bootstrap.runtime import AppRuntime
from app.core.config import Settings
from app.core.lifespan import create_lifespan


def create_app(
    settings: Settings | None = None, runtime: AppRuntime | None = None
) -> FastAPI:
    app_runtime = runtime or AppRuntime(settings or Settings.from_env())
    application = FastAPI(title="CrushPilot", lifespan=create_lifespan(app_runtime))
    application.state.runtime = app_runtime
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
