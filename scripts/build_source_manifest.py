from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge.governance.source_manifest import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument(
        "--chunk-counts",
        type=Path,
        help="JSON object produced by corpus ingestion: {source_path: chunk_count}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "trellis" / "retrieval-system-refactor" / "source-manifest.yaml",
    )
    args = parser.parse_args()
    chunk_counts = (
        json.loads(args.chunk_counts.read_text(encoding="utf-8"))
        if args.chunk_counts
        else None
    )
    write_manifest(build_manifest(args.corpus_version, chunk_counts=chunk_counts), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
