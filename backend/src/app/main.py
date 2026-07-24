from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import register_exception_handlers
from app.api.v1.router import health_router
from app.api.v1.router import router as v1_router
from app.core.config import Settings, get_settings
from app.core.lifespan import create_lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or get_settings()
    application = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        lifespan=create_lifespan(active_settings),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_router)
    application.include_router(
        v1_router,
        prefix=active_settings.api_v1_prefix,
    )
    register_exception_handlers(application)
    return application


app = create_app()
