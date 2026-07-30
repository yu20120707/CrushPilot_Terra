import unittest

from app.knowledge.observability.metrics import REQUIRED_METRICS, RetrievalMetrics
from app.knowledge.observability.retrieval_trace import sanitize_trace_payload


class ObservabilityTests(unittest.TestCase):
    def test_all_required_metrics_are_exposed(self):
        metrics = RetrievalMetrics()
        metrics.observe("selected_chunk_count", 3)
        metrics.observe("trace_write_failure_count", 1)
        snapshot = metrics.snapshot()
        self.assertEqual(set(snapshot), set(REQUIRED_METRICS))
        self.assertEqual(snapshot["selected_chunk_count"]["latest"], 3)

    def test_trace_does_not_store_raw_message_by_default(self):
        payload = sanitize_trace_payload(
            {"request_id": "r", "current_message": "private message"}
        )
        self.assertNotIn("private message", str(payload))
        self.assertEqual(payload["current_message"]["length"], 15)


if __name__ == "__main__":
    unittest.main()
