import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx

os.environ["DEMO_MODE"] = "true"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient

from app.main import (
    ChatResult,
    app,
    call_json,
    contains_unsafe_advice,
    safe_result,
    validate_result,
)


class CrushPilotTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    def setUp(self):
        self.client = TestClient(app)
        self.device = str(uuid4())
        self.headers = {"X-Device-Id": self.device}

    def test_health_readiness_and_metrics(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        readiness = self.client.get("/ready").json()
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["mode"], "demo")
        metrics = self.client.get("/metrics").json()
        self.assertIn("scene_analysis_latency_ms", metrics)
        self.assertIn("trace_write_failure_count", metrics)

    def test_chat_keeps_sse_contract_and_skill(self):
        thread_id = str(uuid4())
        response = self.client.post(
            "/api/v1/chat",
            headers=self.headers,
            json={"thread_id": thread_id, "message": "她说今天很累，怎么回？"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: start", response.text)
        self.assertIn("event: complete", response.text)
        self.assertIn('"skill": "goutoujunshi"', response.text)
        self.assertIn("event: end", response.text)
        self.assertNotIn("event: token", response.text)

    def test_high_risk_shortcut_is_safe_and_still_uses_skill(self):
        response = self.client.post(
            "/api/v1/chat",
            headers=self.headers,
            json={"thread_id": str(uuid4()), "message": "怎么跟踪她"},
        )
        payload_line = next(
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: {") and '"skill"' in line
        )
        payload = json.loads(payload_line)
        self.assertEqual(payload["skill"], "goutoujunshi")
        self.assertIsNotNone(payload["warning"])

    def test_thread_ownership_isolated(self):
        thread_id = str(uuid4())
        self.client.post(
            "/api/v1/chat",
            headers=self.headers,
            json={"thread_id": thread_id, "message": "你好"},
        )
        other = {"X-Device-Id": str(uuid4())}
        self.assertEqual(
            self.client.get(f"/api/v1/threads/{thread_id}", headers=other).status_code,
            404,
        )

    def test_safe_negation_is_allowed_but_executable_advice_is_replaced(self):
        self.assertFalse(contains_unsafe_advice("不要跟踪，也不要骚扰对方。"))
        unsafe = ChatResult(
            intent="测试",
            judgement="判断",
            recommended_reply="你可以跟踪对方。",
            alternatives=["去楼下堵她。", "继续骚扰。"],
        )
        self.assertEqual(
            validate_result(unsafe, "测试").recommended_reply,
            safe_result("测试").recommended_reply,
        )
        unsafe_judgement = unsafe.model_copy(
            update={
                "recommended_reply": "先冷静。",
                "alternatives": ["保持距离。", "尊重边界。"],
                "judgement": "继续纠缠她直到同意。",
                "warning": "建议偷拍取证。",
            }
        )
        self.assertEqual(
            validate_result(unsafe_judgement, "测试").recommended_reply,
            safe_result("测试").recommended_reply,
        )

    def test_model_retries_transport_error_then_succeeds(self):
        schema = ChatResult(
            intent="reply",
            judgement="判断",
            recommended_reply="回复",
            alternatives=["一", "二"],
        )
        request = httpx.Request("POST", "https://model.test")
        success = Mock()
        success.raise_for_status.return_value = None
        success.json.return_value = {
            "choices": [{"message": {"content": schema.model_dump_json()}}]
        }
        with (
            patch("app.main.MODEL_BASE_URL", "https://model.test"),
            patch("app.main.MODEL_API_KEY", "key"),
            patch("app.main.MODEL_NAME", "model"),
            patch(
                "app.main.httpx.post",
                side_effect=[httpx.ConnectError("down", request=request), success],
            ) as post,
            patch("app.main.time.sleep"),
        ):
            self.assertEqual(call_json("system", "user", ChatResult), schema)
        self.assertEqual(post.call_count, 2)

    def test_removed_legacy_runtime_modules_are_absent(self):
        app_dir = Path(__file__).parents[1] / "app"
        self.assertFalse((app_dir / "knowledge_base.py").exists())
        self.assertFalse((app_dir / "skills" / "goutoujunshi.py").exists())


if __name__ == "__main__":
    unittest.main()
