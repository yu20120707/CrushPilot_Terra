import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.research import ResearchAnalysis, analyze_full_corpus, corpus_documents, estimate_corpus_cost


class ResearchCorpusTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "nested").mkdir()
        (self.root / "safe.md").write_text("尊重边界。", encoding="utf-8")
        self.research_content = "研究材料需要完整保留，并且不能被模型作为报告原文回显。"
        (self.root / "nested" / "research.md").write_text(self.research_content, encoding="utf-8")

    def test_corpus_enumerates_every_markdown_document(self):
        self.assertEqual([path.relative_to(self.root).as_posix() for path, _ in corpus_documents(self.root)], ["nested/research.md", "safe.md"])

    def test_full_analysis_injects_each_document_once_and_returns_no_raw_content(self):
        analysis = ResearchAnalysis(risk_level="high", themes=["研究"], harmful_patterns=["操控"], research_summary="研究摘要。")
        with patch("app.research.call_json", return_value=analysis) as mocked_call:
            report = analyze_full_corpus(self.root)
        self.assertEqual(mocked_call.call_count, 2)
        injected = "\n".join(call.args[1] for call in mocked_call.call_args_list)
        self.assertIn("尊重边界。", injected)
        self.assertIn(self.research_content, injected)
        self.assertNotIn("content", report[0])
        self.assertEqual({item["source"] for item in report}, {"safe.md", "nested/research.md"})
        self.assertTrue(all(item["status"] == "ok" for item in report))

    def test_model_excerpt_is_rejected_from_persisted_report(self):
        analysis = ResearchAnalysis(risk_level="high", themes=["研究"], harmful_patterns=[], research_summary=self.research_content)
        with patch("app.research.call_json", return_value=analysis):
            report = analyze_full_corpus(self.root)
        rejected = next(item for item in report if item["source"] == "nested/research.md")
        self.assertEqual(rejected["status"], "output_rejected")
        self.assertNotIn("research_summary", rejected)
        self.assertNotIn(self.research_content, str(rejected))

    def test_failed_document_is_recorded_and_does_not_stop_remaining_documents(self):
        analysis = ResearchAnalysis(risk_level="low", themes=["沟通"], harmful_patterns=[], research_summary="摘要。")
        with patch("app.research.call_json", side_effect=[RuntimeError("network"), analysis]):
            report = analyze_full_corpus(self.root)
        failed = next(item for item in report if item["source"] == "nested/research.md")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertNotIn(self.research_content, str(failed))
        self.assertEqual(next(item for item in report if item["source"] == "safe.md")["status"], "ok")

    def test_checkpoint_resumes_matching_source_without_new_model_request(self):
        checkpoint = self.root / "run.checkpoint.json"
        analysis = ResearchAnalysis(risk_level="low", themes=["沟通"], harmful_patterns=[], research_summary="摘要。")
        with patch("app.research.call_json", return_value=analysis) as first_run:
            first_report = analyze_full_corpus(self.root, checkpoint)
        self.assertEqual(first_run.call_count, 2)
        self.assertTrue(checkpoint.exists())
        with patch("app.research.call_json") as resumed:
            resumed_report = analyze_full_corpus(self.root, checkpoint)
        self.assertEqual(resumed.call_count, 0)
        self.assertEqual(resumed_report, first_report)

    def test_changed_document_is_reanalyzed_while_other_checkpoint_records_resume(self):
        checkpoint = self.root / "run.checkpoint.json"
        analysis = ResearchAnalysis(risk_level="low", themes=["沟通"], harmful_patterns=[], research_summary="摘要。")
        with patch("app.research.call_json", return_value=analysis):
            analyze_full_corpus(self.root, checkpoint)
        (self.root / "safe.md").write_text("更新后的尊重边界。", encoding="utf-8")
        with patch("app.research.call_json", return_value=analysis) as resumed:
            analyze_full_corpus(self.root, checkpoint)
        self.assertEqual(resumed.call_count, 1)

    def test_cost_estimate_is_model_free_and_matches_document_count(self):
        estimate = estimate_corpus_cost(self.root)
        self.assertEqual(estimate["documents"], 2)
        self.assertEqual(estimate["model_requests"], 2)
        self.assertGreater(estimate["estimated_input_tokens"], 0)
        self.assertEqual(estimate["estimated_output_tokens"], 600)


if __name__ == "__main__":
    unittest.main()
