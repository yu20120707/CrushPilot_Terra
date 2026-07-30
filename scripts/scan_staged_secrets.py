from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


TOKEN_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{20,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)
ASSIGNMENT_PATTERN = re.compile(
    r"(?im)(?:api[_-]?key|secret|password)\s*[:=]\s*[\"']?([^\s\"']+)"
)
ENV_REFERENCE = re.compile(r"\$(?:\{[A-Z][A-Z0-9_]*\}|[A-Z][A-Z0-9_]*)\Z")
PLACEHOLDER = re.compile(
    r"(?:change-me|replace-with-(?:your-[a-z0-9-]+|a-long-random-password))\Z",
    re.IGNORECASE,
)


def _allowed_placeholder(value: str) -> bool:
    return bool(ENV_REFERENCE.fullmatch(value) or PLACEHOLDER.fullmatch(value))


def contains_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
        return True
    return any(
        len(value) >= 20
        and not _allowed_placeholder(value)
        for value in ASSIGNMENT_PATTERN.findall(text)
    )


def decode_text(content: bytes) -> str | None:
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16")
    if b"\0" in content:
        return None  # Explicitly skip binary index entries.
    return content.decode("utf-8")


def scan_staged(root: Path) -> list[str]:
    names = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    suspects = []
    for raw_name in names:
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="surrogateescape")
        content = subprocess.run(
            ["git", "show", f":{name}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        try:
            text = decode_text(content)
        except UnicodeDecodeError:
            suspects.append(name)
            continue
        if text is not None and contains_secret(text):
            suspects.append(name)
    return suspects


def main(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    suspects = scan_staged(root)
    print(json.dumps({"potential_secret_paths": suspects}, ensure_ascii=False))
    raise SystemExit(1 if suspects else 0)


if __name__ == "__main__":
    main()
