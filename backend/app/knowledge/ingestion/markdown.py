from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SETEXT = re.compile(r"^\s*(=+|-+)\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: tuple[str, ...]
    content: str
    blocks: tuple["MarkdownBlock", ...]


@dataclass(frozen=True)
class MarkdownBlock:
    kind: str
    text: str


def _blocks(lines: list[str]) -> tuple[MarkdownBlock, ...]:
    result: list[MarkdownBlock] = []
    current: list[str] = []
    open_fence: tuple[str, int] | None = None

    def flush() -> None:
        text = "\n".join(current).strip()
        if not text:
            current.clear()
            return
        first = current[0].lstrip()
        kind = (
            "code"
            if first.startswith(("```", "~~~"))
            else "quote"
            if first.startswith(">")
            else "list"
            if re.match(r"(?:[-*+]|\d+[.)])\s+", first)
            else "table"
            if "|" in first and len(current) > 1 and re.search(r"\|?\s*:?-{3,}", current[1])
            else "paragraph"
        )
        result.append(MarkdownBlock(kind, text))
        current.clear()

    for line in lines:
        fence = FENCE.match(line)
        if fence and open_fence is None:
            flush()
            marker = fence.group(1)
            open_fence = (marker[0], len(marker))
            current.append(line)
            continue
        if (
            fence
            and open_fence is not None
            and fence.group(1)[0] == open_fence[0]
            and len(fence.group(1)) >= open_fence[1]
        ):
            current.append(line)
            open_fence = None
            flush()
            continue
        if not line.strip() and open_fence is None:
            flush()
        else:
            current.append(line)
    flush()
    return tuple(result)


def parse_markdown(text: str, fallback_title: str) -> list[MarkdownSection]:
    """Parse headings without making ingestion depend on a Markdown package."""
    headings: list[str] = []
    sections: list[MarkdownSection] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(
                MarkdownSection(
                    tuple(headings or [fallback_title]), content, _blocks(body)
                )
            )
        body.clear()

    open_fence: tuple[str, int] | None = None
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if open_fence is None:
                open_fence = (marker[0], len(marker))
            elif marker[0] == open_fence[0] and len(marker) >= open_fence[1]:
                open_fence = None
        match = None if open_fence is not None or fence else HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
            headings[level - 1 :] = [title]
        elif (
            open_fence is None
            and not fence
            and index + 1 < len(lines)
            and line.strip()
            and (setext := SETEXT.match(lines[index + 1]))
        ):
            flush()
            level = 1 if setext.group(1).startswith("=") else 2
            headings[level - 1 :] = [line.strip()]
            index += 1
        else:
            body.append(line)
        index += 1
    flush()
    return sections


def parse_markdown_file(path: Path) -> list[MarkdownSection]:
    return parse_markdown(path.read_text(encoding="utf-8"), path.stem)
