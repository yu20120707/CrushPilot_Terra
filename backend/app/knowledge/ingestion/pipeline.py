from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.knowledge.domain.models import KnowledgeChunk
from app.knowledge.governance.source_manifest import (
    build_manifest,
    scan_sources,
    source_root,
    validate_manifest,
)

from .chunker import chunk_sections
from .markdown import parse_markdown_file
from .metadata_enricher import suggest_metadata


SUGGESTED_METADATA_FIELDS = (
    "knowledge_type",
    "topics",
    "applicable_when",
    "not_applicable_when",
    "action_labels",
)
ID_NAMESPACE = uuid.UUID("9ac6cc94-3ddf-5f39-9c5a-e2343e78840b")
ARTIFACT_FILENAMES = {
    "manifest": "source-manifest.yaml",
    "chunk_counts": "chunk-counts.json",
    "build_report": "build-report.json",
}


@dataclass(frozen=True)
class ChunkRelationships:
    parent_section_id: str | None
    previous_chunk_id: str | None
    next_chunk_id: str | None


@dataclass(frozen=True)
class IngestedChunk:
    chunk: KnowledgeChunk
    relationships: ChunkRelationships


@dataclass(frozen=True)
class BuildReport:
    corpus_version: str
    embedding_model: str
    embedding_dimension: int
    embedding_normalization: str
    tokenizer_version: str
    chunker_version: str
    document_count: int
    chunk_count: int
    missing_metadata: dict[str, int]
    missing_metadata_ratios: dict[str, float]


@dataclass(frozen=True)
class CorpusBuild:
    chunks: list[IngestedChunk]
    manifest: dict
    report: BuildReport


def _stable_id(prefix: str, *parts: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, "\0".join((prefix, *parts))))


def ingest_document(
    path: Path, root: Path, corpus_version: str
) -> list[IngestedChunk]:
    relative = path.relative_to(root).as_posix()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    document_id = _stable_id("document", corpus_version, relative, source_hash)
    semantic_chunks = chunk_sections(parse_markdown_file(path))
    chunks: list[KnowledgeChunk] = []
    for index, item in enumerate(semantic_chunks):
        suggested = suggest_metadata(
            item.heading_path[-1], list(item.heading_path), item.content
        )
        parent_path = item.heading_path[:-1]
        parent_id = (
            _stable_id("section", document_id, *parent_path) if parent_path else None
        )
        chunk_id = _stable_id(
            "chunk", document_id, "/".join(item.heading_path), str(index), item.content
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                parent_section_id=parent_id,
                title=item.heading_path[-1],
                heading_path=list(item.heading_path),
                content=item.content,
                knowledge_type=suggested.knowledge_type,
                topics=suggested.topics,
                task_types=[],
                relationship_stages=[],
                action_labels=suggested.action_labels,
                applicable_when=suggested.applicable_when,
                not_applicable_when=suggested.not_applicable_when,
                evidence_level="L4",
                review_status="approved",
                priority=0,
                source_path=relative,
                source_sha256=source_hash,
                corpus_version=corpus_version,
                token_count=item.token_count,
            )
        )
    return [
        IngestedChunk(
            chunk=chunk,
            relationships=ChunkRelationships(
                parent_section_id=chunk.parent_section_id,
                previous_chunk_id=chunks[index - 1].chunk_id if index else None,
                next_chunk_id=(
                    chunks[index + 1].chunk_id
                    if index + 1 < len(chunks)
                    else None
                ),
            ),
        )
        for index, chunk in enumerate(chunks)
    ]


def build_corpus(
    corpus_version: str,
    root: Path | None = None,
    *,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_dimension: int = 512,
    embedding_normalization: str = "l2",
    tokenizer_version: str = "jieba-0.42",
    chunker_version: str = "heading-semantic-v1",
) -> CorpusBuild:
    root = root or source_root()
    by_document = [
        (path, ingest_document(path, root, corpus_version))
        for path in scan_sources(root)
    ]
    counts = {
        path.relative_to(root).as_posix(): len(chunks)
        for path, chunks in by_document
    }
    manifest = build_manifest(corpus_version, chunk_counts=counts, root=root)
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("corpus validation failed: " + "; ".join(errors))
    chunks = [chunk for _, document_chunks in by_document for chunk in document_chunks]
    missing = {
        field: sum(not getattr(record.chunk, field) for record in chunks)
        for field in SUGGESTED_METADATA_FIELDS
    }
    total = len(chunks)
    report = BuildReport(
        corpus_version=corpus_version,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_normalization=embedding_normalization,
        tokenizer_version=tokenizer_version,
        chunker_version=chunker_version,
        document_count=len(by_document),
        chunk_count=total,
        missing_metadata=missing,
        missing_metadata_ratios={
            field: count / total if total else 0.0
            for field, count in missing.items()
        },
    )
    return CorpusBuild(chunks=chunks, manifest=manifest, report=report)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_artifact_paths(output: Path) -> dict[str, Path]:
    """Stable output-path contract shared by ingestion and its caller scripts."""
    return {
        name: output / filename for name, filename in ARTIFACT_FILENAMES.items()
    }


def write_build_artifacts(build: CorpusBuild, output: Path) -> dict[str, Path]:
    chunk_counts = {
        source["path"]: source["chunk_count"]
        for source in build.manifest["sources"]
    }
    report = {
        "corpus_version": build.report.corpus_version,
        "embedding_model": build.report.embedding_model,
        "embedding_dimension": build.report.embedding_dimension,
        "embedding_normalization": build.report.embedding_normalization,
        "tokenizer_version": build.report.tokenizer_version,
        "chunker_version": build.report.chunker_version,
        "document_count": build.report.document_count,
        "chunk_count": build.report.chunk_count,
        "missing_metadata": build.report.missing_metadata,
        "missing_metadata_ratios": build.report.missing_metadata_ratios,
    }
    artifacts = build_artifact_paths(output)
    _atomic_write(
        artifacts["manifest"],
        yaml.safe_dump(build.manifest, allow_unicode=True, sort_keys=False),
    )
    _atomic_write(
        artifacts["chunk_counts"],
        json.dumps(chunk_counts, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        artifacts["build_report"],
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return artifacts
