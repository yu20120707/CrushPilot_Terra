from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


ALLOWED_SOURCE_GROUPS = ("knowledge", "practical")
ALLOWED_STATUSES = {"ingested", "approved_excluded"}
FORBIDDEN_RELEASE_STATUSES = {"pending", "unknown", "sample_only", "later"}


def source_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "goutoujunshi_source"
        / "references"
    )


def scan_sources(root: Path | None = None) -> list[Path]:
    root = root or source_root()
    return sorted(
        path
        for group in ALLOWED_SOURCE_GROUPS
        for path in (root / group).rglob("*.md")
        if path.is_file()
    )


def build_manifest(
    corpus_version: str,
    chunk_counts: dict[str, int] | None = None,
    excluded_paths: set[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or source_root()
    chunk_counts = chunk_counts or {}
    excluded_paths = excluded_paths or set()
    sources = []
    for index, path in enumerate(scan_sources(root), 1):
        relative = path.relative_to(root).as_posix()
        excluded = relative in excluded_paths
        chunk_count = chunk_counts.get(relative, 0)
        sources.append(
            {
                "source_id": f"KB-{index:03d}",
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "required": not excluded,
                "status": (
                    "approved_excluded"
                    if excluded
                    else "ingested"
                    if chunk_count > 0
                    else "pending"
                ),
                "chunk_count": 0 if excluded else chunk_count,
            }
        )
    return {
        "corpus_version": corpus_version,
        "source_root": root.as_posix(),
        "sources": sources,
    }


def validate_manifest(manifest: dict[str, Any], root: Path | None = None) -> list[str]:
    root = root or source_root()
    errors: list[str] = []
    actual = {path.relative_to(root).as_posix() for path in scan_sources(root)}
    entries = manifest.get("sources", [])
    paths = [entry.get("path") for entry in entries]
    if set(paths) != actual or len(paths) != len(set(paths)):
        errors.append("manifest must contain every source exactly once")
    for entry in entries:
        status = entry.get("status")
        if status not in ALLOWED_STATUSES or status in FORBIDDEN_RELEASE_STATUSES:
            errors.append(f"{entry.get('path')}: invalid release status {status!r}")
        path = root / str(entry.get("path", ""))
        if not path.is_file():
            errors.append(f"{entry.get('path')}: source is missing")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            errors.append(f"{entry.get('path')}: source hash mismatch")
        if entry.get("required") and int(entry.get("chunk_count", 0)) <= 0:
            errors.append(f"{entry.get('path')}: required source has zero chunks")
    return errors


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid source manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("source manifest root must be an object")
    return manifest
