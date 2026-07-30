from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "hardcoded_knowledge_routing": re.compile(
        r"\b(?:selected_reference_paths|reference_context|TERM_ALIASES)\b"
    ),
    "json_vector_index": re.compile(
        r"knowledge_vectors\.json|\bretrieve_by_vector\b|json\.loads?\([^)]*vector",
        re.IGNORECASE,
    ),
    "str_count_retrieval": re.compile(r"\.count\(\s*(?:term|keyword)", re.IGNORECASE),
    "full_markdown_prompt": re.compile(
        r"GOUTOUJUNSHI_REFERENCES|reference_context\([^)]*\).*prompt",
        re.IGNORECASE,
    ),
    "duplicate_skill_prompt": re.compile(
        r"\b(?:RUNTIME_INSTRUCTIONS|CHAT_SYSTEM_PROMPT)\b"
    ),
}


def detect(scan_root: Path) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    for path in sorted(scan_root.rglob("*.py")):
        if any(part in {"tests", "__pycache__", ".venv"} for part in path.parts):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for category, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "category": category,
                            "path": path.relative_to(scan_root).as_posix(),
                            "line": line_number,
                        }
                    )
    return {
        "legacy_retrieval_paths": len(findings),
        "details": {"findings": findings},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-root", type=Path, default=ROOT / "backend/app"
    )
    args = parser.parse_args()
    try:
        result = detect(args.scan_root)
    except (OSError, UnicodeError) as error:
        result = {
            "legacy_retrieval_paths": 1,
            "details": {"error": str(error)},
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(1 if result["legacy_retrieval_paths"] else 0)


if __name__ == "__main__":
    main()
