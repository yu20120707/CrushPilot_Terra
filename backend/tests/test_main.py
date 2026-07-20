import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ["DEMO_MODE"] = "true"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient
from app.main import KNOWLEDGE_DIR, app, model_result, retrieve_knowledge, route_skill


class CrushPilotTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_routes_four_skills(self):
        cases = [("她说今天累了怎么回", "reply-suggestion"), ("短一点", "reply-rewrite"), ("她回复慢怎么办", "cold-recovery"), ("周末怎么约她", "date-invitation")]
        for message, expected in cases:
            self.assertEqual(route_skill({"user_message": message})["current_skill"], expected)

    def test_chat_sse_and_thread_recovery(self):
        response = self.client.post("/api/v1/chat", json={"thread_id": "ut-thread", "message": "她说刚下班很累，我怎么回？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: start", response.text)
        self.assertIn("event: complete", response.text)
        self.assertIn("event: end", response.text)
        thread = self.client.get("/api/v1/threads/ut-thread")
        self.assertEqual(thread.status_code, 200)
        self.assertGreaterEqual(len(thread.json()["messages"]), 2)

    def test_safety_warning(self):
        response = self.client.post("/api/v1/chat", json={"thread_id": "safe-thread", "message": "怎么跟踪她"})
        self.assertIn("不提供骚扰", response.text)

    def test_server_error_does_not_expose_internal_details(self):
        with patch("app.main.graph.invoke", side_effect=RuntimeError("internal model endpoint detail")):
            response = self.client.post("/api/v1/chat", json={"thread_id": "error-thread", "message": "怎么回"})
        self.assertIn("服务暂时不可用", response.text)
        self.assertNotIn("internal model endpoint detail", response.text)

    def test_local_knowledge_cards(self):
        self.assertTrue((KNOWLEDGE_DIR / "reply-suggestion.md").exists())
        result = retrieve_knowledge({"current_skill": "date-invitation"})
        self.assertIn("低压力", result["retrieved_knowledge"][0]["content"])

    @patch("app.main.httpx.post")
    def test_model_receives_recent_dialogue(self, mocked_post):
        mocked_post.return_value.raise_for_status.return_value = None
        mocked_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"skill":"reply-suggestion","judgement":"接住话题。","recommended_reply":"辛苦了。","alternatives":["早点休息。","明天再聊。"],"warning":null}'}}]
        }
        with patch("app.main.DEMO_MODE", False), patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "test-key"), patch("app.main.MODEL_NAME", "test-model"):
            model_result(
                "reply-suggestion",
                "我该怎么回？",
                "先回应情绪。",
                [{"role": "user", "content": "她说今天很累"}, {"role": "assistant", "content": "那早点休息"}, {"role": "user", "content": "我该怎么回？"}],
            )
        prompt = mocked_post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("她说今天很累", prompt)
        self.assertIn("那早点休息", prompt)

    def test_delete_thread(self):
        thread_id = "delete-thread"
        self.client.post("/api/v1/chat", json={"thread_id": thread_id, "message": "怎么回"})
        self.assertIn(thread_id, [item["thread_id"] for item in self.client.get("/api/v1/threads").json()])
        self.assertEqual(self.client.delete(f"/api/v1/threads/{thread_id}").json(), {"deleted": True})
        self.assertEqual(self.client.get(f"/api/v1/threads/{thread_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
