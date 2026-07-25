from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    loader = json.loads if path.suffix == ".json" else yaml.safe_load
    value = loader(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def verify(manifest_path: Path, report_path: Path, expected_path: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    report = _load(report_path)
    expected = _load(expected_path)
    mismatches: list[str] = []

    for artifact_name, artifact in (("manifest", manifest), ("build_report", report)):
        if artifact.get("corpus_version") != expected.get("corpus_version"):
            mismatches.append(f"{artifact_name}.corpus_version")
    for field in ("embedding_model", "tokenizer_version", "chunker_version"):
        if report.get(field) != expected.get(field) or not expected.get(field):
            mismatches.append(f"build_report.{field}")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        mismatches.append("manifest.sources")
        sources = []
    if any(source.get("status") != "ingested" for source in sources):
        mismatches.append("manifest.source_status")
    if any(
        not isinstance(source.get("chunk_count"), int)
        or source["chunk_count"] <= 0
        for source in sources
    ):
        mismatches.append("manifest.source_chunk_count")
    chunk_total = sum(
        source.get("chunk_count", 0)
        for source in sources
        if isinstance(source.get("chunk_count"), int)
    )
    if report.get("document_count") != len(sources):
        mismatches.append("build_report.document_count")
    if report.get("chunk_count") != chunk_total or chunk_total <= 0:
        mismatches.append("build_report.chunk_count")

    return {
        "corpus_version_mismatches": len(set(mismatches)),
        "details": {"mismatched_fields": sorted(set(mismatches))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    directory = ROOT / "trellis/retrieval-system-refactor"
    parser.add_argument(
        "--manifest", type=Path, default=directory / "source-manifest.yaml"
    )
    parser.add_argument(
        "--build-report", type=Path, default=directory / "build-report.json"
    )
    parser.add_argument(
        "--expected", type=Path, default=directory / "corpus-version.json"
    )
    args = parser.parse_args()
    try:
        result = verify(args.manifest, args.build_report, args.expected)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as error:
        result = {
            "corpus_version_mismatches": 1,
            "details": {"error": str(error)},
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(1 if result["corpus_version_mismatches"] else 0)


if __name__ == "__main__":
    main()
