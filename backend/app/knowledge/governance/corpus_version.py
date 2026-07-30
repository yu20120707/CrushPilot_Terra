from __future__ import annotations

from dataclasses import dataclass


CURRENT_CORPUS_VERSION = "2026.08.1"


@dataclass(frozen=True)
class CorpusVersion:
    version: str
    embedding_model: str
    embedding_dimension: int
    embedding_normalization: str
    tokenizer_version: str
    chunker_version: str


@dataclass(frozen=True)
class CorpusReadiness:
    ready: bool
    lexical_only: bool = False
    reason: str | None = None
