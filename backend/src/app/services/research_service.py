"""Offline full-corpus research analysis; never used by the user chat workflow."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.assistant.prompts import SYSTEM_PROMPT
from app.core.config import get_settings
from app.infrastructure.model_factory import create_model_client

KNOWLEDGE_DIR = get_settings().knowledge_dir
MODEL_CLIENT = create_model_client(get_settings())


def call_json[SchemaT: BaseModel](
    system: str,
    user: str,
    schema: type[SchemaT],
) -> SchemaT:
    return MODEL_CLIENT.call_json(system, user, schema)


class ResearchAnalysis(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    themes: list[str] = Field(min_length=1, max_length=6)
    harmful_patterns: list[str] = Field(default_factory=list, max_length=6)
    research_summary: str = Field(min_length=1, max_length=500)


def contains_source_excerpt(value: str, source: str, minimum_length: int = 24) -> bool:
    return any(source[index:index + minimum_length] in value for index in range(0, max(0, len(source) - minimum_length + 1)))


def report_record(path: Path, content: str, root: Path, analysis: ResearchAnalysis) -> dict[str, object]:
    fields = [*analysis.themes, *analysis.harmful_patterns, analysis.research_summary]
    base = {"source": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
    if any(contains_source_excerpt(field, content) for field in fields):
        return {**base, "status": "output_rejected"}
    return {**base, "status": "ok", **analysis.model_dump()}


def corpus_documents(root: Path = KNOWLEDGE_DIR) -> list[tuple[Path, str]]:
    return [(path, path.read_text(encoding="utf-8")) for path in sorted(root.rglob("*.md"))]


def estimate_corpus_cost(root: Path = KNOWLEDGE_DIR) -> dict[str, int]:
    """Return a deliberately conservative, model-agnostic request estimate."""
    documents = corpus_documents(root)
    input_characters = sum(len(content) for _, content in documents)
    return {
        "documents": len(documents),
        "model_requests": len(documents),
        "estimated_input_tokens": (input_characters + 3) // 4,
        "estimated_output_tokens": len(documents) * 300,
    }


def load_checkpoint(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("研究 checkpoint 必须是 JSON 列表")
    return {
        str(record["source"]): record
        for record in payload
        if isinstance(record, dict) and record.get("status") == "ok" and isinstance(record.get("source"), str)
    }


def save_checkpoint(path: Path | None, records: list[dict[str, object]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def analyze_full_corpus(root: Path = KNOWLEDGE_DIR, checkpoint_path: Path | None = None) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    completed = load_checkpoint(checkpoint_path)
    prompt = SYSTEM_PROMPT + "\n你在做离线研究语料标注，不向用户提供建议。识别风险与主题，不复述操作步骤，不输出可执行的操控、骚扰、欺骗、性越界或暴力建议。必须返回 JSON 字段：risk_level(low/medium/high), themes, harmful_patterns, research_summary。"
    for path, content in corpus_documents(root):
        relative_path = path.relative_to(root).as_posix()
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        previous = completed.get(relative_path)
        if previous is not None and previous.get("sha256") == source_hash:
            results.append(previous)
            continue
        try:
            analysis = call_json(prompt, f"<RESEARCH_DOCUMENT path='{relative_path}'>{content}</RESEARCH_DOCUMENT>", ResearchAnalysis)
            record = report_record(path, content, root, analysis)
        except Exception as error:
            # Keep one bad document from discarding an expensive completed run.
            record = {"source": relative_path, "sha256": source_hash, "status": "failed", "error_type": type(error).__name__}
        results.append(record)
        save_checkpoint(checkpoint_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze every knowledge Markdown file for offline research.")
    parser.add_argument("--output", required=True, type=Path, help="JSON report path; source documents remain unchanged.")
    parser.add_argument("--checkpoint", type=Path, help="Resume file; defaults to OUTPUT.checkpoint.json.")
    parser.add_argument("--estimate", action="store_true", help="Print request/token estimate without calling the model.")
    args = parser.parse_args()
    if args.estimate:
        print(json.dumps(estimate_corpus_cost(), ensure_ascii=False))
        return
    checkpoint = args.checkpoint or args.output.with_suffix(args.output.suffix + ".checkpoint.json")
    report = analyze_full_corpus(checkpoint_path=checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已处理 {len(report)} 份文档：{args.output}（checkpoint：{checkpoint}）")


if __name__ == "__main__":
    main()
