import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ["DEMO_MODE"] = "true"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient
from app.main import ChatResult, KNOWLEDGE_DIR, MAX_SSE_BYTES, MAX_SSE_LINE, app, model_result, retrieve_knowledge, route_skill, stream_model_result, validate_result


class CrushPilotTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-Device-Id": "c544979f-8d93-4c16-aa79-c330aa51ee65"}

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_routes_four_skills(self):
        cases = [("她说今天累了怎么回", "reply-suggestion"), ("短一点", "reply-rewrite"), ("她回复慢怎么办", "cold-recovery"), ("周末怎么约她", "date-invitation")]
        for message, expected in cases:
            self.assertEqual(route_skill({"user_message": message})["current_skill"], expected)

    def test_chat_sse_and_thread_recovery(self):
        response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "ut-thread", "message": "她说刚下班很累，我怎么回？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: start", response.text)
        self.assertIn("event: complete", response.text)
        self.assertIn("event: end", response.text)
        thread = self.client.get("/api/v1/threads/ut-thread", headers=self.headers)
        self.assertEqual(thread.status_code, 200)
        self.assertEqual(len(thread.json()["messages"]), 2)

        self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "ut-thread", "message": "那要不要明天再聊？"})
        self.assertEqual(len(self.client.get("/api/v1/threads/ut-thread", headers=self.headers).json()["messages"]), 4)

    def test_safety_warning(self):
        response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "safe-thread", "message": "怎么跟踪她"})
        self.assertIn("不提供骚扰", response.text)

    def test_server_error_does_not_expose_internal_details(self):
        with patch("app.main.graph.invoke", side_effect=RuntimeError("internal model endpoint detail")):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "error-thread", "message": "怎么回"})
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
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["role"], "system")
        prompt = payload["messages"][1]["content"]
        self.assertIn("她说今天很累", prompt)
        self.assertIn("那早点休息", prompt)

    def test_delete_thread(self):
        thread_id = "delete-thread"
        self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": thread_id, "message": "怎么回"})
        self.assertIn(thread_id, [item["thread_id"] for item in self.client.get("/api/v1/threads", headers=self.headers).json()])
        self.assertEqual(self.client.delete(f"/api/v1/threads/{thread_id}", headers=self.headers).json(), {"deleted": True})
        self.assertEqual(self.client.get(f"/api/v1/threads/{thread_id}", headers=self.headers).status_code, 404)

    def test_threads_are_device_private(self):
        owner = {"X-Device-Id": "c544979f-8d93-4c16-aa79-c330aa51ee65"}
        stranger = {"X-Device-Id": "467c1c91-65b3-4f47-a9f7-f45c4321ac87"}
        self.client.post("/api/v1/chat", headers=owner, json={"thread_id": "private-thread", "message": "怎么回"})
        self.assertEqual(self.client.get("/api/v1/threads/private-thread", headers=stranger).status_code, 404)
        self.assertEqual(self.client.delete("/api/v1/threads/private-thread", headers=stranger).status_code, 404)
        self.assertEqual(self.client.post("/api/v1/chat", headers=stranger, json={"thread_id": "private-thread", "message": "继续聊"}).status_code, 404)
        self.assertNotIn("private-thread", [item["thread_id"] for item in self.client.get("/api/v1/threads", headers=stranger).json()])

    def test_history_keeps_a_twelve_message_window(self):
        for index in range(7):
            self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "window-thread", "message": f"怎么回 {index}"})
        history = self.client.get("/api/v1/threads/window-thread", headers=self.headers).json()["messages"]
        self.assertEqual(len(history), 12)

    def test_invalid_device_id_is_rejected(self):
        response = self.client.post("/api/v1/chat", headers={"X-Device-Id": "not-a-uuid"}, json={"thread_id": "bad-device", "message": "怎么回"})
        self.assertEqual(response.status_code, 400)

    def test_model_output_is_checked_and_skill_is_fixed(self):
        unsafe = ChatResult(skill="date-invitation", judgement="正常", recommended_reply="你可以威胁她。", alternatives=["继续施压。", "别放弃。"])
        result = validate_result(unsafe, "reply-suggestion")
        self.assertEqual(result.skill, "reply-suggestion")
        self.assertIsNotNone(result.warning)

        safe = ChatResult(skill="date-invitation", judgement="正常", recommended_reply="给对方选择。", alternatives=["周末有空吗？", "不方便也没关系。"])
        self.assertEqual(validate_result(safe, "reply-suggestion").skill, "reply-suggestion")

    @patch("app.main.httpx.stream")
    def test_model_streams_and_parses_result(self, mocked_stream):
        response = mocked_stream.return_value.__enter__.return_value
        response.raise_for_status.return_value = None
        chunks = [
            '{"recommended_reply":"早',
            '点休息","skill":"reply-suggestion","judgement":"接住情绪。","alternatives":["辛苦了。","明天再聊。"],"warning":null}',
        ]
        response.iter_bytes.return_value = [f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]}, ensure_ascii=False)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]
        with patch("app.main.DEMO_MODE", False), patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "test-key"), patch("app.main.MODEL_NAME", "test-model"):
            events = list(stream_model_result("reply-suggestion", "怎么回", "先回应。", [{"role": "user", "content": "她很累"}]))
        self.assertEqual("".join(token for token, _ in events), "")
        self.assertEqual(events[-1][1].recommended_reply, "早点休息")

    @patch("app.main.httpx.stream")
    def test_model_stream_rejects_oversized_line_before_json_parse(self, mocked_stream):
        response = mocked_stream.return_value.__enter__.return_value
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [b"data: " + b"x" * MAX_SSE_LINE]
        with patch("app.main.DEMO_MODE", False), patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "test-key"), patch("app.main.MODEL_NAME", "test-model"):
            with self.assertRaisesRegex(RuntimeError, "SSE 行过大"):
                list(stream_model_result("reply-suggestion", "怎么回", "先回应。", [{"role": "user", "content": "她很累"}]))

    @patch("app.main.httpx.stream")
    def test_model_stream_rejects_excessive_total_bytes(self, mocked_stream):
        response = mocked_stream.return_value.__enter__.return_value
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [b"event: ping\n" * (MAX_SSE_BYTES // 12 + 1)]
        with patch("app.main.DEMO_MODE", False), patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "test-key"), patch("app.main.MODEL_NAME", "test-model"):
            with self.assertRaisesRegex(RuntimeError, "SSE 响应过大"):
                list(stream_model_result("reply-suggestion", "怎么回", "先回应。", [{"role": "user", "content": "她很累"}]))

    def test_production_stream_persists_once(self):
        answer = ChatResult(skill="reply-suggestion", judgement="接住情绪。", recommended_reply="早点休息。", alternatives=["辛苦了。", "明天再聊。"])
        with patch("app.main.DEMO_MODE", False), patch("app.main.stream_model_result", return_value=iter([("早点", None), ("休息。", None), ("", answer)])):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "stream-thread", "message": "她很累"})
        self.assertIn("早点", response.text)
        self.assertEqual(len(self.client.get("/api/v1/threads/stream-thread", headers=self.headers).json()["messages"]), 2)

    def test_production_stream_checks_output_before_sending_tokens(self):
        unsafe = ChatResult(skill="reply-suggestion", judgement="正常", recommended_reply="你可以威胁她。", alternatives=["继续施压。", "别放弃。"])
        with patch("app.main.DEMO_MODE", False), patch("app.main.stream_model_result", return_value=iter([("你可以威胁她。", None), ("", unsafe)])):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "safe-stream-thread", "message": "她很累"})
        self.assertNotIn("你可以威胁她", response.text)
        self.assertIn("尊重对方", response.text)


if __name__ == "__main__":
    unittest.main()
