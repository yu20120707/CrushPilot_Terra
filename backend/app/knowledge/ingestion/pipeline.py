from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

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
from .metadata_enricher import SuggestedMetadata, suggest_metadata


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
    metadata_suggestion_status: Literal["accepted", "missing", "invalid"]


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
    metadata_suggestion_statuses: dict[str, int]
    metadata_suggestion_ratios: dict[str, float]
    collection_document_counts: dict[str, int]
    collection_chunk_counts: dict[str, int]
    usage_scope_counts: dict[str, int]
    review_status_counts: dict[str, int]


@dataclass(frozen=True)
class CorpusBuild:
    chunks: list[IngestedChunk]
    manifest: dict
    report: BuildReport


def _stable_id(prefix: str, *parts: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, "\0".join((prefix, *parts))))


def _admission_for_new_kb(title: str, heading_path: list[str], content: str) -> tuple[str, str, str]:
    """Return usage scope, review status and an auditable deterministic reason."""
    text = " ".join((title, *heading_path, content)).lower()
    # New-library rejection is now reserved for explicit, audited overrides.
    # Generic keyword matching produced false positives and must not auto-reject chunks.
    unsafe_markers: tuple[str, ...] = ()
    safe_markers = ("尊重", "共情", "边界", "同意", "沟通", "倾听", "道歉", "拒绝", "留空间", "不施压", "健康")
    if any(marker in text for marker in unsafe_markers):
        return "research_only", "rejected", "规则隔离：命中操控、伤害或明显不适合在线建议的风险词；等待人工复核上下文"
    # Keyword signals are useful for routing but insufficient evidence for a public
    # release.  New material stays out of ordinary chat until an auditable review
    # explicitly changes both fields in staging.
    if any(marker in text for marker in safe_markers):
        return "research_only", "draft", "规则候选：命中健康沟通信号，但新来源必须经人工准入后才能在线使用"
    return "research_only", "draft", "规则隔离：缺少可复查的在线准入信号，等待人工审核"


def _decision_key(title: str, topics: list[str]) -> str:
    """Use a narrow, reviewable semantic decision label rather than broad topics.

    Topics are retrieval facets, not mutually exclusive conclusions.  Treating a
    whole topic set as one decision would suppress unrelated advice across source
    collections.  Matching headings provide a conservative default; reviewers can
    explicitly align a decision key when recording a true supersession.
    """
    return title.strip().lower()


def _manual_admission_overrides(root: Path) -> dict[str, dict[str, object]]:
    path = root / "admission-overrides.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("chunks", data) if isinstance(data, dict) else {}
    if not isinstance(entries, dict):
        raise ValueError("admission-overrides.yaml must map source paths to overrides")
    return entries


def ingest_document(
    path: Path,
    root: Path,
    corpus_version: str,
    metadata_suggester: Callable[[str, list[str], str], object | None] | None = None,
    *,
    source_collection: Literal["original", "new_kb"] = "original",
    admission_overrides: dict[str, dict[str, object]] | None = None,
) -> list[IngestedChunk]:
    relative = path.relative_to(root).as_posix()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    document_id = _stable_id("document", corpus_version, relative, source_hash)
    semantic_chunks = chunk_sections(parse_markdown_file(path))
    chunks: list[tuple[KnowledgeChunk, Literal["accepted", "missing", "invalid"]]] = []
    for index, item in enumerate(semantic_chunks):
        fallback = suggest_metadata(
            item.heading_path[-1], list(item.heading_path), item.content
        )
        suggested = fallback
        suggestion_status: Literal["accepted", "missing", "invalid"] = "missing"
        if metadata_suggester is not None:
            try:
                raw_suggestion = metadata_suggester(
                    item.heading_path[-1], list(item.heading_path), item.content
                )
                if raw_suggestion is not None:
                    suggested = SuggestedMetadata.model_validate(raw_suggestion)
                    suggestion_status = "accepted"
            except Exception:
                suggestion_status = "invalid"
        parent_path = item.heading_path[:-1]
        parent_id = (
            _stable_id("section", document_id, *parent_path) if parent_path else None
        )
        chunk_id = _stable_id(
            "chunk", document_id, "/".join(item.heading_path), str(index), item.content
        )
        usage_scope, review_status, admission_reason = (
            _admission_for_new_kb(item.heading_path[-1], list(item.heading_path), item.content)
            if source_collection == "new_kb"
            else ("online_eligible", "approved", "已发布原库：沿用既有审核准入")
        )
        override = (admission_overrides or {}).get(relative) or (admission_overrides or {}).get("*")
        if override is not None:
            usage_scope = str(override.get("usage_scope", usage_scope))
            review_status = str(override.get("review_status", review_status))
            admission_reason = str(override.get("admission_reason", admission_reason))
            if (usage_scope, review_status) != ("online_eligible", "approved") and usage_scope != "research_only":
                raise ValueError(f"{relative}: invalid admission override")
        chunks.append((
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
                review_status=review_status,
                usage_scope=usage_scope,
                source_collection=source_collection,
                source_priority=100 if source_collection == "new_kb" else 50,
                decision_key=_decision_key(item.heading_path[-1], suggested.topics),
                stance=item.content[:240],
                admission_reason=admission_reason,
                priority=0,
                source_path=relative,
                source_sha256=source_hash,
                corpus_version=corpus_version,
                token_count=item.token_count,
            ),
            suggestion_status,
        ))
    return [
        IngestedChunk(
            chunk=chunk,
            relationships=ChunkRelationships(
                parent_section_id=chunk.parent_section_id,
                previous_chunk_id=chunks[index - 1][0].chunk_id if index else None,
                next_chunk_id=(
                    chunks[index + 1][0].chunk_id
                    if index + 1 < len(chunks)
                    else None
                ),
            ),
            metadata_suggestion_status=status,
        )
        for index, (chunk, status) in enumerate(chunks)
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
    metadata_suggester: Callable[[str, list[str], str], object | None] | None = None,
    new_kb_root: Path | None = None,
) -> CorpusBuild:
    root = root or source_root()
    if new_kb_root is not None and (not new_kb_root.is_dir()):
        raise ValueError(f"new knowledge root is missing: {new_kb_root}")
    original_documents = [
        ("original", path, root, ingest_document(path, root, corpus_version, metadata_suggester))
        for path in scan_sources(root)
    ]
    new_documents = []
    if new_kb_root is not None:
        new_kb_root = new_kb_root.resolve()
        admission_overrides = _manual_admission_overrides(new_kb_root)
        new_documents = [
            ("new_kb", path, new_kb_root, ingest_document(path, new_kb_root, corpus_version, metadata_suggester, source_collection="new_kb", admission_overrides=admission_overrides))
            for path in sorted(new_kb_root.rglob("*.md")) if path.is_file()
        ]
    if new_kb_root is not None and not new_documents:
        raise ValueError(f"new knowledge root has no Markdown sources: {new_kb_root}")
    by_document = original_documents + new_documents
    counts = {
        path.relative_to(document_root).as_posix(): len(chunks)
        for collection, path, document_root, chunks in original_documents
    }
    manifest = build_manifest(corpus_version, chunk_counts=counts, root=root)
    manifest["collections"] = {
        "original": {"source_root": root.as_posix(), "document_count": len(original_documents)},
        "new_kb": {"source_root": new_kb_root.as_posix(), "document_count": len(new_documents)} if new_kb_root else None,
    }
    errors = validate_manifest(manifest, root)
    if errors:
        raise ValueError("corpus validation failed: " + "; ".join(errors))
    chunks = [chunk for _, _, _, document_chunks in by_document for chunk in document_chunks]
    missing = {
        field: sum(not getattr(record.chunk, field) for record in chunks)
        for field in SUGGESTED_METADATA_FIELDS
    }
    total = len(chunks)
    suggestion_statuses = {
        status: sum(
            record.metadata_suggestion_status == status for record in chunks
        )
        for status in ("accepted", "missing", "invalid")
    }
    report = BuildReport(
        corpus_version=corpus_version,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_normalization=embedding_normalization,
        tokenizer_version=tokenizer_version,
        chunker_version=chunker_version,
        document_count=len(by_document),
        chunk_count=total,
        collection_document_counts={collection: sum(1 for item in by_document if item[0] == collection) for collection in ("original", "new_kb")},
        collection_chunk_counts={collection: sum(1 for item in chunks if item.chunk.source_collection == collection) for collection in ("original", "new_kb")},
        usage_scope_counts={scope: sum(1 for item in chunks if item.chunk.usage_scope == scope) for scope in ("online_eligible", "research_only")},
        review_status_counts={status: sum(1 for item in chunks if item.chunk.review_status == status) for status in ("approved", "draft", "rejected", "deprecated")},
        missing_metadata=missing,
        missing_metadata_ratios={
            field: count / total if total else 0.0
            for field, count in missing.items()
        },
        metadata_suggestion_statuses=suggestion_statuses,
        metadata_suggestion_ratios={
            status: count / total if total else 0.0
            for status, count in suggestion_statuses.items()
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
        "metadata_suggestion_statuses": build.report.metadata_suggestion_statuses,
        "metadata_suggestion_ratios": build.report.metadata_suggestion_ratios,
        "collection_document_counts": build.report.collection_document_counts,
        "collection_chunk_counts": build.report.collection_chunk_counts,
        "usage_scope_counts": build.report.usage_scope_counts,
        "review_status_counts": build.report.review_status_counts,
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
