from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge.governance.source_manifest import read_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "trellis" / "retrieval-system-refactor" / "source-manifest.yaml",
    )
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print("unprocessed_sources=0")


if __name__ == "__main__":
    main()
