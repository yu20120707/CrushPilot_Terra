from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.governance.corpus_version import CURRENT_CORPUS_VERSION


def _enabled(value: str | None) -> bool:
    return (value or "false").lower() == "true"


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    knowledge_dir: Path
    skill_dir: Path
    demo_mode: bool
    local_sqlite_mode: bool
    database_url: str
    model_provider: str
    model_base_url: str
    model_api_key: str
    model_name: str
    model_trust_env: bool
    trace_debug: bool
    embedding_model: str
    embedding_dimension: int
    embedding_normalization: str
    tokenizer_version: str
    chunker_version: str
    reranker_model: str
    corpus_version: str
    allow_lexical_only: bool

    @classmethod
    def from_env(cls) -> "Settings":
        # 配置层只读取环境变量和推导路径；连接、模型探测等 I/O 由 AppRuntime 管理，
        # 避免导入模块时产生不可控副作用。
        root = Path(__file__).resolve().parents[3]
        app_dir = Path(__file__).resolve().parents[1]
        provider = os.getenv("MODEL_PROVIDER", "custom").lower()
        prefix = {"deepseek": "DEEPSEEK", "longxia": "LONGXIA"}.get(provider, "MODEL")
        return cls(
            root=root,
            data_dir=Path(os.getenv("DATA_DIR", root / "data")),
            knowledge_dir=root / "knowledge",
            skill_dir=Path(
                os.getenv(
                    "SKILL_DIR",
                    app_dir / "agents" / "assistant" / "resources" / "goutoujunshi_skill",
                )
            ),
            demo_mode=_enabled(os.getenv("DEMO_MODE")),
            local_sqlite_mode=_enabled(os.getenv("LOCAL_SQLITE_MODE")),
            database_url=os.getenv("DATABASE_URL", ""),
            model_provider=provider,
            model_base_url=os.getenv(
                f"{prefix}_BASE_URL", os.getenv("MODEL_BASE_URL", "")
            ).rstrip("/"),
            model_api_key=os.getenv(f"{prefix}_API_KEY", os.getenv("MODEL_API_KEY", "")),
            model_name=os.getenv(f"{prefix}_MODEL", os.getenv("MODEL_NAME", "")),
            model_trust_env=_enabled(os.getenv("MODEL_TRUST_ENV", "true")),
            trace_debug=_enabled(os.getenv("TRACE_DEBUG")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "512")),
            embedding_normalization=os.getenv("EMBEDDING_NORMALIZATION", "l2"),
            tokenizer_version=os.getenv("TOKENIZER_VERSION", "jieba-0.42"),
            chunker_version=os.getenv("CHUNKER_VERSION", "heading-semantic-v1"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
            corpus_version=os.getenv("CORPUS_VERSION", CURRENT_CORPUS_VERSION),
            allow_lexical_only=_enabled(os.getenv("ALLOW_LEXICAL_ONLY_STARTUP")),
        )
