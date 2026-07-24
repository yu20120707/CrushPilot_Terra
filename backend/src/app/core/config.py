from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_origins(value: str | None) -> tuple[str, ...]:
    defaults = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    )
    if not value:
        return defaults
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


def _project_paths() -> tuple[Path, Path, Path]:
    backend_root = Path(__file__).resolve().parents[3]
    local_knowledge = backend_root / "knowledge"
    repository_knowledge = backend_root.parent / "knowledge"
    if local_knowledge.is_dir():
        return backend_root, backend_root, local_knowledge
    return backend_root, backend_root.parent, repository_knowledge


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    app_version: str
    api_v1_prefix: str
    demo_mode: bool
    model_provider: str
    model_base_url: str
    model_api_key: str
    model_name: str
    database_url: str
    embedding_model: str
    backend_root: Path
    repository_root: Path
    knowledge_dir: Path
    data_dir: Path
    cors_origins: tuple[str, ...]
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        backend_root, repository_root, default_knowledge_dir = _project_paths()
        provider = os.getenv("MODEL_PROVIDER", "custom").strip().lower()
        provider_prefix = {
            "deepseek": "DEEPSEEK",
            "longxia": "LONGXIA",
        }.get(provider, "MODEL")
        base_url = os.getenv(
            f"{provider_prefix}_BASE_URL",
            os.getenv("MODEL_BASE_URL", ""),
        ).rstrip("/")
        api_key = os.getenv(
            f"{provider_prefix}_API_KEY",
            os.getenv("MODEL_API_KEY", ""),
        )
        model_name = os.getenv(
            f"{provider_prefix}_MODEL",
            os.getenv("MODEL_NAME", ""),
        )
        return cls(
            app_name=os.getenv("APP_NAME", "CrushPilot"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
            demo_mode=_as_bool(os.getenv("DEMO_MODE"), default=True),
            model_provider=provider,
            model_base_url=base_url,
            model_api_key=api_key,
            model_name=model_name,
            database_url=os.getenv("DATABASE_URL", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            backend_root=backend_root,
            repository_root=repository_root,
            knowledge_dir=Path(os.getenv("KNOWLEDGE_DIR", default_knowledge_dir)),
            data_dir=Path(os.getenv("DATA_DIR", repository_root / "data")),
            cors_origins=_as_origins(os.getenv("CORS_ORIGINS")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
