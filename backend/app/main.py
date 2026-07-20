import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "").rstrip("/")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
KNOWLEDGE_DIR = ROOT / "knowledge"


Skill = Literal["reply-suggestion", "reply-rewrite", "cold-recovery", "date-invitation"]


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class ChatResult(BaseModel):
    skill: Skill
    judgement: str = Field(min_length=1, max_length=500)
    recommended_reply: str = Field(min_length=1, max_length=1000)
    alternatives: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(min_length=2, max_length=2)
    warning: str | None = Field(default=None, max_length=500)


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], lambda old, new: (old + new)[-12:]]
    user_message: str
    device_id: str
    current_skill: str
    retrieved_knowledge: list[dict[str, str]]
    final_response: dict[str, Any]


SKILLS = {
    "reply-suggestion": {"keywords": ["怎么回", "回复", "她说", "他说"], "judgement": "对方在分享状态，先接住情绪和话题。"},
    "reply-rewrite": {"keywords": ["自然一点", "短一点", "暧昧", "不要提问", "换一个", "改"], "judgement": "保持原意，按你的风格要求改写。"},
    "cold-recovery": {"keywords": ["冷", "回复慢", "冷场", "不回", "已读"], "judgement": "先降低压力，不要连续追问或强行推进。"},
    "date-invitation": {"keywords": ["约", "周末", "见面", "邀约", "出来"], "judgement": "邀约应延续已有话题，给对方低压力选择。"},
}
UNSAFE_PATTERN = re.compile(r"跟踪|尾随|蹲守|堵(?:她|他|人|门|路)|威胁|恐吓|强迫|逼迫|纠缠|骚扰|未成年|骗(?:她|他)|诈骗|冒充|假装身份|常去.{0,8}(?:等|堵)|连续.{0,8}(?:联系|发消息)")
SYSTEM_PROMPT = "你是恋爱聊天助手。不得提供骚扰、威胁、跟踪、强迫、欺骗、冒充或未成年人相关建议。只返回 JSON，字段为 recommended_reply, skill, judgement, alternatives(恰好两条), warning。"
MAX_MODEL_JSON = 12_000
MAX_SSE_LINE = 48_000
MAX_SSE_BYTES = 64_000


def safe_result(skill: Skill) -> ChatResult:
    return ChatResult(skill=skill, judgement="这类做法可能伤害对方或越过边界。", recommended_reply="先尊重对方的意愿和边界，不要继续施压。", alternatives=["如果对方不想继续，请给彼此一点空间。", "先冷静下来，再用尊重的方式沟通。"], warning="不提供骚扰、威胁、跟踪、欺骗、冒充或未成年人相关建议。")


def contains_unsafe_content(text: str) -> bool:
    return bool(UNSAFE_PATTERN.search(text))


def validate_result(result: ChatResult, skill: Skill) -> ChatResult:
    content = "\n".join([result.judgement, result.recommended_reply, *result.alternatives])
    if contains_unsafe_content(content):
        return safe_result(skill)
    return result.model_copy(update={"skill": skill})


def route_skill(state: ChatState) -> ChatState:
    text = state["user_message"].lower()
    for skill in ("reply-rewrite", "cold-recovery", "date-invitation", "reply-suggestion"):
        definition = SKILLS[skill]
        if any(word in text for word in definition["keywords"]):
            return {"current_skill": skill}
    return {"current_skill": "reply-suggestion"}


def retrieve_knowledge(state: ChatState) -> ChatState:
    skill = state["current_skill"]
    card = (KNOWLEDGE_DIR / f"{skill}.md").read_text(encoding="utf-8")
    return {"retrieved_knowledge": [{"skill": skill, "content": card[:1200]}]}


def resolve_context(state: ChatState) -> ChatState:
    return {}


def model_prompt(message: str, knowledge: str, recent_messages: list[dict[str, str]]) -> str:
    dialogue = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
        for item in recent_messages[-12:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    return f"知识：{knowledge}\n最近对话：\n{dialogue}\n当前用户输入：{message}"


def model_payload(message: str, knowledge: str, recent_messages: list[dict[str, str]], stream: bool = False) -> dict[str, Any]:
    payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": model_prompt(message, knowledge, recent_messages)}], "response_format": {"type": "json_object"}}
    if stream:
        payload["stream"] = True
    return payload


def model_result(skill: Skill, message: str, knowledge: str, recent_messages: list[dict[str, str]]) -> ChatResult:
    if contains_unsafe_content("\n".join(item.get("content", "") for item in recent_messages)):
        return safe_result(skill)
    if DEMO_MODE:
        samples = {
            "reply-suggestion": ("听起来你今天挺累的，回去早点休息。", ["辛苦了，今晚适合什么都不干。", "那先好好充电，明天再慢慢聊。"]),
            "reply-rewrite": ("那你先好好休息，别太累着。", ["今天辛苦啦，早点躺平。", "先去休息吧，晚点再聊。"]),
            "cold-recovery": ("最近是不是有点忙？不用急着回，空了再聊。", ["我看到这个还挺想起你，忙完再说。", "这两天先各忙各的，之后再聊。"]),
            "date-invitation": ("你之前提过那家店，周六下午要不要一起去试试？", ["周末如果你有空，我们去喝杯咖啡？", "这家展览你可能会喜欢，要不要一起看看？"]),
        }
        reply, alternatives = samples[skill]
        return ChatResult(skill=skill, judgement=SKILLS[skill]["judgement"], recommended_reply=reply, alternatives=alternatives)
    if not (MODEL_BASE_URL and MODEL_API_KEY and MODEL_NAME):
        raise RuntimeError("模型未配置：请设置 MODEL_BASE_URL、MODEL_API_KEY 和 MODEL_NAME，或只在本地开启 DEMO_MODE=true。")
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    payload = model_payload(message, knowledge, recent_messages)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = httpx.post(f"{MODEL_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return ChatResult.model_validate_json(content)
        except Exception as error:  # ponytail: a single retry covers transient JSON/model failures.
            last_error = error
    raise RuntimeError(f"模型响应不可用：{last_error}")


def stream_model_result(skill: Skill, message: str, knowledge: str, recent_messages: list[dict[str, str]]):
    if contains_unsafe_content("\n".join(item.get("content", "") for item in recent_messages)):
        yield "", safe_result(skill)
        return
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    payload = model_payload(message, knowledge, recent_messages, stream=True)
    content = ""
    with httpx.stream("POST", f"{MODEL_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=20) as response:
        response.raise_for_status()
        buffer = b""
        total_sse_bytes = 0
        for chunk in response.iter_bytes(chunk_size=1024):
            total_sse_bytes += len(chunk)
            if total_sse_bytes > MAX_SSE_BYTES:
                raise RuntimeError("模型 SSE 响应过大")
            buffer += chunk
            if len(buffer) > MAX_SSE_LINE:
                raise RuntimeError("模型 SSE 行过大")
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                if len(raw_line) > MAX_SSE_LINE:
                    raise RuntimeError("模型 SSE 行过大")
                if not raw_line.startswith(b"data: "):
                    continue
                data = raw_line[6:]
                if data == b"[DONE]":
                    buffer = b""
                    break
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                if len(content) + len(delta) > MAX_MODEL_JSON:
                    raise RuntimeError("模型响应过大")
                content += delta
    yield "", ChatResult.model_validate_json(content)


def generate_reply(state: ChatState) -> ChatState:
    knowledge = state["retrieved_knowledge"][0]["content"]
    result = model_result(state["current_skill"], state["user_message"], knowledge, state.get("messages", []))
    result = validate_result(result, state["current_skill"])
    return {"final_response": result.model_dump(), "messages": [{"role": "assistant", "content": result.recommended_reply}]}


def format_response(state: ChatState) -> ChatState:
    return {}


def save_context(state: ChatState) -> ChatState:
    return {}


def build_graph() -> tuple[Any, Any]:
    if DATABASE_URL.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        global postgres_context
        postgres_context = PostgresSaver.from_conn_string(DATABASE_URL)
        checkpointer = postgres_context.__enter__()
    else:
        connection = sqlite3.connect(DATA_DIR / "checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    graph = StateGraph(ChatState)
    graph.add_node("resolve_context", resolve_context)
    graph.add_node("route_skill", route_skill)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("generate_reply", generate_reply)
    graph.add_node("format_response", format_response)
    graph.add_node("save_context", save_context)
    graph.add_edge(START, "resolve_context")
    graph.add_edge("resolve_context", "route_skill")
    graph.add_edge("route_skill", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "generate_reply")
    graph.add_edge("generate_reply", "format_response")
    graph.add_edge("format_response", "save_context")
    graph.add_edge("save_context", END)
    return graph.compile(checkpointer=checkpointer), checkpointer


graph, checkpointer = build_graph()
app = FastAPI(title="CrushPilot")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def device_id(x_device_id: str = Header(alias="X-Device-Id")) -> str:
    try:
        return str(UUID(x_device_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="无效设备标识") from error


def owned_state(thread_id: str, owner_id: str) -> Any:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not state.values or state.values.get("device_id") != owner_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return state


@app.post("/api/v1/chat")
def chat(request: ChatRequest, owner_id: str = Depends(device_id)) -> StreamingResponse:
    existing = graph.get_state({"configurable": {"thread_id": request.thread_id}})
    if existing.values and existing.values.get("device_id") != owner_id:
        raise HTTPException(status_code=404, detail="会话不存在")

    def events():
        yield "event: start\ndata: {}\n\n"
        try:
            config = {"configurable": {"thread_id": request.thread_id}}
            if DEMO_MODE:
                result = graph.invoke({"user_message": request.message, "device_id": owner_id, "messages": [{"role": "user", "content": request.message}]}, config)
                answer = result["final_response"]
                reply = answer["recommended_reply"]
                for index in range(0, len(reply), 8):
                    yield f"event: token\ndata: {json.dumps({'text': reply[index:index + 8]}, ensure_ascii=False)}\n\n"
            else:
                prior_messages = existing.values.get("messages", []) if existing.values else []
                current_skill = route_skill({"user_message": request.message})["current_skill"]
                knowledge = retrieve_knowledge({"current_skill": current_skill})["retrieved_knowledge"][0]["content"]
                answer = None
                for token, streamed_result in stream_model_result(current_skill, request.message, knowledge, [*prior_messages, {"role": "user", "content": request.message}]):
                    if streamed_result:
                        answer = validate_result(streamed_result, current_skill).model_dump()
                if answer is None:
                    raise RuntimeError("模型流未返回结果")
                for index in range(0, len(answer["recommended_reply"]), 8):
                    yield f"event: token\ndata: {json.dumps({'text': answer['recommended_reply'][index:index + 8]}, ensure_ascii=False)}\n\n"
                graph.update_state(config, {"user_message": request.message, "device_id": owner_id, "current_skill": current_skill, "retrieved_knowledge": [{"skill": current_skill, "content": knowledge}], "final_response": answer, "messages": [{"role": "user", "content": request.message}, {"role": "assistant", "content": answer["recommended_reply"]}]}, as_node="save_context")
            yield f"event: complete\ndata: {json.dumps(answer, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用，请稍后重试。'}, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/v1/threads")
def list_threads(owner_id: str = Depends(device_id)) -> list[dict[str, str]]:
    threads: list[dict[str, str]] = []
    thread_ids = {checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)}
    for thread_id in thread_ids:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        if state.values.get("device_id") == owner_id:
            first_message = next((item["content"] for item in state.values.get("messages", []) if item["role"] == "user"), thread_id)
            threads.append({"thread_id": thread_id, "title": first_message[:30]})
    return threads


@app.get("/api/v1/threads/{thread_id}")
def get_thread(thread_id: str, owner_id: str = Depends(device_id)) -> dict[str, Any]:
    state = owned_state(thread_id, owner_id)
    return {"thread_id": thread_id, "messages": state.values.get("messages", [])}


@app.delete("/api/v1/threads/{thread_id}")
def delete_thread(thread_id: str, owner_id: str = Depends(device_id)) -> dict[str, bool]:
    owned_state(thread_id, owner_id)
    checkpointer.delete_thread(thread_id)
    return {"deleted": True}
