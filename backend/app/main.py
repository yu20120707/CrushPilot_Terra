import json
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Any, TypedDict

import httpx
from fastapi import FastAPI, HTTPException
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


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class ChatResult(BaseModel):
    skill: str
    judgement: str
    recommended_reply: str
    alternatives: list[str] = Field(min_length=2, max_length=2)
    warning: str | None = None


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], lambda old, new: old + new]
    user_message: str
    current_skill: str
    retrieved_knowledge: list[dict[str, str]]
    final_response: dict[str, Any]


SKILLS = {
    "reply-suggestion": {"keywords": ["怎么回", "回复", "她说", "他说"], "judgement": "对方在分享状态，先接住情绪和话题。"},
    "reply-rewrite": {"keywords": ["自然一点", "短一点", "暧昧", "不要提问", "换一个", "改"], "judgement": "保持原意，按你的风格要求改写。"},
    "cold-recovery": {"keywords": ["冷", "回复慢", "冷场", "不回", "已读"], "judgement": "先降低压力，不要连续追问或强行推进。"},
    "date-invitation": {"keywords": ["约", "周末", "见面", "邀约", "出来"], "judgement": "邀约应延续已有话题，给对方低压力选择。"},
}
BLOCKED = ("跟踪", "威胁", "逼她", "逼他", "未成年", "骗她", "骗他")


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
    return state


def model_result(skill: str, message: str, knowledge: str, recent_messages: list[dict[str, str]]) -> ChatResult:
    if any(word in message for word in BLOCKED):
        return ChatResult(skill=skill, judgement="这类做法可能伤害对方或越过边界。", recommended_reply="先尊重对方的意愿和边界，不要继续施压。", alternatives=["如果对方不想继续，请给彼此一点空间。", "先冷静下来，再用尊重的方式沟通。"], warning="不提供骚扰、威胁、跟踪、欺骗或未成年人相关建议。")
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
    dialogue = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
        for item in recent_messages[-12:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    prompt = f"你是恋爱聊天助手。知识：{knowledge}\n最近对话：\n{dialogue}\n当前用户输入：{message}\n仅返回 JSON，字段为 skill, judgement, recommended_reply, alternatives(恰好两条), warning。"
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
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


def generate_reply(state: ChatState) -> ChatState:
    knowledge = state["retrieved_knowledge"][0]["content"]
    result = model_result(state["current_skill"], state["user_message"], knowledge, state.get("messages", []))
    return {"final_response": result.model_dump(), "messages": [{"role": "assistant", "content": result.recommended_reply}]}


def format_response(state: ChatState) -> ChatState:
    return state


def save_context(state: ChatState) -> ChatState:
    return state


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


@app.post("/api/v1/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    def events():
        yield "event: start\ndata: {}\n\n"
        try:
            config = {"configurable": {"thread_id": request.thread_id}}
            result = graph.invoke({"user_message": request.message, "messages": [{"role": "user", "content": request.message}]}, config)
            answer = result["final_response"]
            reply = answer["recommended_reply"]
            for index in range(0, len(reply), 8):
                yield f"event: token\ndata: {json.dumps({'text': reply[index:index + 8]}, ensure_ascii=False)}\n\n"
            yield f"event: complete\ndata: {json.dumps(answer, ensure_ascii=False)}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用，请稍后重试。'}, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/v1/threads")
def list_threads() -> list[dict[str, str]]:
    seen: set[str] = set()
    threads: list[dict[str, str]] = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.add(thread_id)
            threads.append({"thread_id": thread_id, "title": thread_id})
    return threads


@app.get("/api/v1/threads/{thread_id}")
def get_thread(thread_id: str) -> dict[str, Any]:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not state.values:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"thread_id": thread_id, "messages": state.values.get("messages", [])}


@app.delete("/api/v1/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, bool]:
    checkpointer.delete_thread(thread_id)
    return {"deleted": True}
