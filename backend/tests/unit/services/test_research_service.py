from unittest.mock import Mock

from app.services import research_service
from app.services.research_service import (
    ResearchAnalysis,
    analyze_full_corpus,
    corpus_documents,
    estimate_corpus_cost,
)


def create_corpus(root):
    (root / "nested").mkdir()
    (root / "safe.md").write_text("尊重边界。", encoding="utf-8")
    content = "研究材料需要完整保留，并且不能被模型作为报告原文回显。"
    (root / "nested" / "research.md").write_text(content, encoding="utf-8")
    return content


def test_corpus_analysis_is_resumable_and_does_not_persist_source(
    tmp_path,
    monkeypatch,
) -> None:
    source = create_corpus(tmp_path)
    analysis = ResearchAnalysis(
        risk_level="low",
        themes=["沟通"],
        harmful_patterns=[],
        research_summary="摘要。",
    )
    model_call = Mock(return_value=analysis)
    monkeypatch.setattr(research_service, "call_json", model_call)
    checkpoint = tmp_path / "run.checkpoint.json"

    first = analyze_full_corpus(tmp_path, checkpoint)
    assert model_call.call_count == 2
    assert source not in str(first)

    model_call.reset_mock()
    resumed = analyze_full_corpus(tmp_path, checkpoint)
    assert model_call.call_count == 0
    assert resumed == first


def test_model_excerpt_is_rejected(tmp_path, monkeypatch) -> None:
    source = create_corpus(tmp_path)
    analysis = ResearchAnalysis(
        risk_level="high",
        themes=["研究"],
        harmful_patterns=[],
        research_summary=source,
    )
    monkeypatch.setattr(research_service, "call_json", Mock(return_value=analysis))

    report = analyze_full_corpus(tmp_path)
    rejected = next(item for item in report if item["source"] == "nested/research.md")
    assert rejected["status"] == "output_rejected"
    assert "research_summary" not in rejected
    assert source not in str(rejected)


def test_cost_estimate_is_model_free(tmp_path) -> None:
    create_corpus(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path, _ in corpus_documents(tmp_path)] == [
        "nested/research.md",
        "safe.md",
    ]
    estimate = estimate_corpus_cost(tmp_path)
    assert estimate["documents"] == 2
    assert estimate["model_requests"] == 2
    assert estimate["estimated_output_tokens"] == 600
