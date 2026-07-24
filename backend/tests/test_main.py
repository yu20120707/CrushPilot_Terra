import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

os.environ["DEMO_MODE"] = "true"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient
from app.main import (
    ChatResult,
    CHAT_SYSTEM_PROMPT,
    IntentAnalysis,
    KNOWLEDGE_DIR,
    SAFE_CARD_IDS,
    SAFE_CARD_INDEX,
    app,
    build_knowledge_index,
    call_json,
    contains_unsafe_advice,
    retrieve_by_keywords,
    retrieve_knowledge,
    safe_result,
    should_refine,
    validate_result,
)
from app.knowledge_base import RAW_KNOWLEDGE_DIR, build_research_cards, build_research_vector_index, read_cards, research_manifest, retrieve_by_vector
from app.skills.goutoujunshi import SOURCE_ROOT, UPSTREAM_COMMIT, reference_context, selected_reference_paths
from app.thread_store import ThreadStore


class CrushPilotTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)

    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-Device-Id": "c544979f-8d93-4c16-aa79-c330aa51ee65"}

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_raw_knowledge_is_read_only_and_every_document_has_research_provenance(self):
        before = {path: path.read_bytes() for path in KNOWLEDGE_DIR.rglob("*.md")}
        build_knowledge_index()
        self.assertEqual(before, {path: path.read_bytes() for path in KNOWLEDGE_DIR.rglob("*.md")})
        self.assertEqual(len(research_manifest()), len(list(RAW_KNOWLEDGE_DIR.rglob("*.md"))))
        self.assertEqual({card["id"] for card in read_cards()}, SAFE_CARD_IDS)
        self.assertEqual({chunk["source"] for chunk in SAFE_CARD_INDEX}, {f"card:{card_id}" for card_id in SAFE_CARD_IDS})

    def test_safe_card_retrieval_covers_income_and_single_character_fatigue(self):
        income = retrieve_by_keywords(["工资", "经济条件"])
        fatigue = retrieve_by_keywords(["累"])
        self.assertEqual(income[0]["source"], "card:income-demeaning")
        self.assertEqual(fatigue[0]["source"], "card:low-pressure-chat")

    def test_vector_retrieval_is_off_until_a_local_index_exists(self):
        self.assertEqual(retrieve_by_vector("她嫌我工资低", index_path=Path(tempfile.mkdtemp()) / "missing.json"), [])

    def test_every_raw_document_has_a_research_card_and_embedding(self):
        directory = Path(tempfile.mkdtemp())
        try:
            cards = build_research_cards(directory / "cards.json")
            index = build_research_vector_index(directory / "vectors.json", encode_fn=lambda texts: [[float(index)] for index, _ in enumerate(texts)], cards_path=directory / "cards.json")
            self.assertEqual(len(cards), len(list(RAW_KNOWLEDGE_DIR.rglob("*.md"))))
            self.assertEqual(len(index["cards"]), len(cards))
            self.assertEqual(index["scope"], "research_only")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_chat_retrieval_never_returns_raw_research_material(self):
        card = read_cards()[0]
        with patch("app.main.retrieve_by_vector", return_value=[card]):
            hits = retrieve_knowledge({"user_message": "她嫌我工资低", "retrieval_queries": ["工资"]})["retrieved_knowledge"]
        self.assertTrue(hits)
        self.assertTrue(all(hit["source"].startswith("card:") for hit in hits))
        self.assertNotIn("pua-knowledge-sharing", "\n".join(hit["content"] for hit in hits))

    def test_alias_retrieval_handles_fatigue_without_exact_word(self):
        index = [{"source": "reply-suggestion.md", "title": "回复", "content": "听起来今天挺累的，回去早点休息。", "tier": "safe"}]
        self.assertEqual(retrieve_by_keywords(["疲惫"], index), index)

    def test_no_hit_requests_one_refine_then_generates(self):
        self.assertEqual(should_refine({"retrieved_knowledge": [], "retrieval_attempt": 0}), "refine")
        self.assertEqual(should_refine({"retrieved_knowledge": [], "retrieval_attempt": 1}), "generate")

    def test_safe_negation_is_not_rejected_but_executable_advice_is(self):
        safe = ChatResult(intent="ignored", judgement="不要骚扰或强迫对方。", recommended_reply="你可以先表达关心。", alternatives=["辛苦了，早点休息。", "有空再聊。"])
        self.assertIsNone(validate_result(safe, "聊天回应").warning)
        unsafe = ChatResult(intent="ignored", judgement="正常", recommended_reply="你可以威胁她。", alternatives=["继续施压。", "别放弃。"])
        self.assertIsNotNone(validate_result(unsafe, "聊天回应").warning)
        self.assertFalse(contains_unsafe_advice("不要强迫或骚扰对方。"))
        for advice in ["每天在她公司楼下等她", "别停，反复给她发消息直到回应", "灌醉后再表白", "假装偶遇来接近她", "偷拍视频留作把柄"]:
            self.assertTrue(contains_unsafe_advice(advice), advice)

    def test_unknown_model_risk_flag_does_not_become_refusal(self):
        analysis = IntentAnalysis(scene="回应疲惫", goal="低压力回应", features=["疲惫"], keywords=["疲惫"], risk_flags=["explanatory_boundary_note"])
        result = ChatResult(intent="ignored", judgement="共情。", recommended_reply="听起来挺累的，早点休息。", alternatives=["辛苦啦。", "忙完再聊。"])
        evidence = [{"source": "reply-suggestion.md", "title": "回复", "content": "尊重边界", "tier": "safe"}]
        with patch("app.main.DEMO_MODE", False), patch("app.main.call_json", side_effect=[analysis, result]), patch("app.main.retrieve_by_keywords", return_value=evidence):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "risk-flag-thread", "message": "她说累了"})
        self.assertIn("听起来挺累", response.text)

    def test_production_no_hit_rewrites_once(self):
        analysis = IntentAnalysis(scene="回应疲惫", goal="低压力回应", features=["疲惫"], keywords=["疲惫"], risk_flags=[])
        result = ChatResult(intent="ignored", judgement="共情。", recommended_reply="辛苦了，早点休息。", alternatives=["先放松一下。", "忙完再聊。"])
        evidence = [{"source": "reply-suggestion.md", "title": "回复", "content": "尊重边界", "tier": "safe"}]
        with patch("app.main.DEMO_MODE", False), patch("app.main.call_json", side_effect=[analysis, result]) as model, patch("app.main.retrieve_by_keywords", side_effect=[[], evidence]) as retrieve:
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "rewrite-thread", "message": "她很疲惫"})
        self.assertIn("辛苦了", response.text)
        self.assertEqual(model.call_count, 2)
        self.assertEqual([call.args[0] for call in retrieve.call_args_list], [["疲惫"], ["疲惫", "累", "辛苦", "休息"]])

    def test_production_hit_uses_two_model_calls_and_sse_has_no_fake_tokens(self):
        analysis = IntentAnalysis(scene="回应疲惫", goal="低压力回应", features=["疲惫"], keywords=["累"], risk_flags=[])
        result = ChatResult(intent="ignored", judgement="共情。", recommended_reply="辛苦了，早点休息。", alternatives=["先放松一下。", "忙完再聊。"])
        evidence = [{"source": "reply-suggestion.md", "title": "回复", "content": "尊重边界", "tier": "safe"}]
        with patch("app.main.DEMO_MODE", False), patch("app.main.call_json", side_effect=[analysis, result]) as model, patch("app.main.retrieve_by_keywords", return_value=evidence):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "two-call-thread", "message": "她很累"})
        self.assertEqual(model.call_count, 2)
        self.assertIn("event: complete", response.text)
        self.assertNotIn("event: token", response.text)

    def test_goutoujunshi_skill_is_used_for_final_chat_generation(self):
        analysis = IntentAnalysis(scene="回复疲惫", goal="低压力回应", features=["疲惫"], keywords=["累"], risk_flags=[])
        result = ChatResult(intent="ignored", judgement="共情。", recommended_reply="今天辛苦了，先好好休息，等你有空再聊。", alternatives=["先照顾好自己，忙完再说。", "不用急着回，休息好了再聊。"])
        evidence = [{"source": "card:low-pressure-chat", "title": "低压力聊天", "content": "尊重边界", "tier": "safe"}]
        with patch("app.main.DEMO_MODE", False), patch("app.main.call_json", side_effect=[analysis, result]) as model, patch("app.main.retrieve_by_keywords", return_value=evidence):
            self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "goutoujunshi-thread", "message": "她说今天很累"})
        self.assertEqual(model.call_args_list[1].args[0], CHAT_SYSTEM_PROMPT)
        self.assertIn("狗头军师", CHAT_SYSTEM_PROMPT)
        self.assertIn("不得提供贬低", CHAT_SYSTEM_PROMPT)
        final_prompt = model.call_args_list[1].args[1]
        self.assertIn("<GOUTOUJUNSHI_REFERENCES>", final_prompt)
        self.assertIn("关系投入失衡：互惠判断、降级投入与退出决策", final_prompt)
        self.assertIn('"skill": "goutoujunshi"', self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "goutoujunshi-response-thread", "message": "怎么回复"}).text)

    def test_goutoujunshi_marks_the_safety_shortcut_too(self):
        result = safe_result()
        self.assertEqual(result.skill, "goutoujunshi")
        self.assertIn("狗头军师判断", result.judgement)
        response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "goutoujunshi-safety-thread", "message": "怎么跟踪她"})
        self.assertIn('"skill": "goutoujunshi"', response.text)
        self.assertIn("狗头军师判断", response.text)

    def test_complete_goutoujunshi_source_is_vendored_with_noncommercial_notice(self):
        self.assertEqual(UPSTREAM_COMMIT, "81d99da388da11cb6c1bc18259c6802c82fbaf41")
        self.assertTrue((SOURCE_ROOT / "LICENSE").is_file())
        self.assertIn("PolyForm Noncommercial", (SOURCE_ROOT / "LICENSE").read_text(encoding="utf-8"))
        self.assertEqual(len(list((SOURCE_ROOT / "references" / "knowledge").glob("*.md"))), 20)
        self.assertEqual(len(list((SOURCE_ROOT / "references" / "practical").glob("*.md"))), 20)
        self.assertEqual(len(list((SOURCE_ROOT / "tests").glob("*.md"))), 6)

    def test_goutoujunshi_loads_only_routed_references_into_the_prompt(self):
        paths = selected_reference_paths("她说你这点工资够干啥呀", limit=3)
        self.assertEqual([path.name for path in paths], ["12-金钱家务育儿与双方家庭.md", "06-吸引约会与关系启动.md"])
        references = reference_context("她说你这点工资够干啥呀")
        self.assertIn("12-金钱家务育儿与双方家庭.md", references)
        self.assertNotIn("01-证据分级与内容边界.md", references)

    @patch("app.main.time.sleep")
    @patch("app.main.httpx.post")
    def test_model_retries_transport_error_then_succeeds(self, post, _sleep):
        good = Mock()
        good.raise_for_status.return_value = None
        good.json.return_value = {"choices": [{"message": {"content": '{"scene":"回应","goal":"回复","features":["聊天"],"keywords":["沟通"],"risk_flags":[]}'}}]}
        post.side_effect = [httpx.ConnectError("offline"), good]
        with patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "key"), patch("app.main.MODEL_NAME", "model"):
            self.assertEqual(call_json("system", "user", IntentAnalysis).scene, "回应")
        self.assertEqual(post.call_count, 2)

    @patch("app.main.httpx.post")
    def test_model_does_not_retry_client_error(self, post):
        request = httpx.Request("POST", "https://model.example/chat/completions")
        response = httpx.Response(400, request=request)
        bad = Mock()
        bad.raise_for_status.side_effect = httpx.HTTPStatusError("bad", request=request, response=response)
        post.return_value = bad
        with patch("app.main.MODEL_BASE_URL", "https://model.example"), patch("app.main.MODEL_API_KEY", "key"), patch("app.main.MODEL_NAME", "model"):
            with self.assertRaisesRegex(RuntimeError, "模型响应不可用"):
                call_json("system", "user", IntentAnalysis)
        self.assertEqual(post.call_count, 1)

    def test_sse_error_hides_internal_detail(self):
        with patch("app.main.graph.invoke", side_effect=RuntimeError("internal endpoint detail")):
            response = self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": "error-thread", "message": "怎么回"})
        self.assertIn("服务暂时不可用", response.text)
        self.assertNotIn("internal endpoint detail", response.text)

    def test_thread_store_claim_is_atomic_between_owners(self):
        directory = Path(tempfile.mkdtemp())
        try:
            store = ThreadStore("", directory)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda owner: store.claim("same-thread", owner, "title"), ["owner-a", "owner-b"]))
            self.assertEqual(sorted(results), [False, True])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_device_private_thread_lifecycle(self):
        thread_id = "private-thread"
        stranger = {"X-Device-Id": "467c1c91-65b3-4f47-a9f7-f45c4321ac87"}
        self.client.post("/api/v1/chat", headers=self.headers, json={"thread_id": thread_id, "message": "怎么回"})
        self.assertEqual(self.client.get(f"/api/v1/threads/{thread_id}", headers=stranger).status_code, 404)
        self.assertEqual(self.client.post("/api/v1/chat", headers=stranger, json={"thread_id": thread_id, "message": "继续聊"}).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/v1/threads/{thread_id}", headers=self.headers).json(), {"deleted": True})


if __name__ == "__main__":
    unittest.main()
