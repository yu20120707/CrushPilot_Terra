from __future__ import annotations

import re
from dataclasses import dataclass

from .markdown import MarkdownSection


TOKEN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
BLOCK_BREAK = re.compile(r"\n\s*\n")


def token_count(text: str) -> int:
    return len(TOKEN.findall(text))


def _split_long(text: str, target: int, overlap: int) -> list[str]:
    matches = list(TOKEN.finditer(text))
    if len(matches) <= target:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(matches):
        end = min(start + target, len(matches))
        char_start = matches[start].start()
        char_end = matches[end - 1].end()
        chunks.append(text[char_start:char_end].strip())
        if end == len(matches):
            break
        start = end - overlap
    return chunks


def _units(content: str, max_tokens: int, overlap: int) -> list[str]:
    units: list[str] = []
    for block in BLOCK_BREAK.split(content):
        block = block.strip()
        if not block:
            continue
        units.extend(_split_long(block, max_tokens, overlap))
    return units


@dataclass(frozen=True)
class SemanticChunk:
    heading_path: tuple[str, ...]
    content: str
    token_count: int


def chunk_sections(
    sections: list[MarkdownSection],
    *,
    target_min: int = 250,
    target_max: int = 600,
    max_tokens: int = 900,
    short_threshold: int = 120,
    overlap: int = 80,
) -> list[SemanticChunk]:
    if not 60 <= overlap <= 100:
        raise ValueError("overlap must be between 60 and 100 tokens")
    result: list[SemanticChunk] = []
    for section in sections:
        pending = ""
        for unit in _units(section.content, target_max, overlap):
            candidate = f"{pending}\n\n{unit}".strip() if pending else unit
            size = token_count(candidate)
            if pending and size > target_max and token_count(pending) >= target_min:
                result.append(
                    SemanticChunk(section.heading_path, pending, token_count(pending))
                )
                pending = unit
            elif size <= max_tokens:
                pending = candidate
            else:
                if pending:
                    result.append(
                        SemanticChunk(
                            section.heading_path, pending, token_count(pending)
                        )
                    )
                pending = unit
        if pending:
            current = SemanticChunk(
                section.heading_path, pending, token_count(pending)
            )
            if (
                result
                and current.token_count < short_threshold
                and result[-1].heading_path == current.heading_path
                and result[-1].token_count + current.token_count <= max_tokens
            ):
                previous = result.pop()
                merged = f"{previous.content}\n\n{current.content}"
                current = SemanticChunk(
                    current.heading_path, merged, token_count(merged)
                )
            result.append(current)
    return result
