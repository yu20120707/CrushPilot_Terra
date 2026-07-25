from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field, ValidationError

from .agents.assistant.graph import build_assistant_graph
from .agents.assistant.nodes import AssistantNodes, contains_unsafe_advice, safe_result, validate_result
from .agents.assistant.prompts import SAFETY_PROMPT
from .agents.assistant.schemas import ChatResult
from .knowledge.domain.models import EvidenceAssessment, RetrievalTrace
from .knowledge.governance.corpus_version import CorpusVersion
from .knowledge.observability.metrics import RetrievalMetrics
from .knowledge.repositories.postgres import PostgresKnowledgeRepository
from .knowledge.retrieval.lexical_retriever import LexicalRetriever
from .knowledge.retrieval.reranker import RerankerUnavailable
from .knowledge.retrieval.service import RetrievalResult, RetrievalService
from .knowledge.retrieval.vector_retriever import VectorRetriever, VectorUnavailable
from .knowledge.skill.runtime import SkillRuntime
from .thread_store import ThreadStore

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "knowledge"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "custom").lower()
_PROVIDER_PREFIX = {"deepseek": "DEEPSEEK", "longxia": "LONGXIA"}.get(MODEL_PROVIDER, "MODEL")
MODEL_BASE_URL = os.getenv(f"{_PROVIDER_PREFIX}_BASE_URL", os.getenv("MODEL_BASE_URL", "")).rstrip("/")
MODEL_API_KEY = os.getenv(f"{_PROVIDER_PREFIX}_API_KEY", os.getenv("MODEL_API_KEY", ""))
MODEL_NAME = os.getenv(f"{_PROVIDER_PREFIX}_MODEL", os.getenv("MODEL_NAME", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "2026.07.5")
ALLOW_LEXICAL_ONLY = (
    os.getenv("ALLOW_LEXICAL_ONLY_STARTUP", "false").lower() == "true"
)
SKILL_DIR = Path(
    os.getenv(
        "SKILL_DIR",
        Path(__file__).parent / "agents" / "assistant" / "resources" / "goutoujunshi_skill",
    )
)
class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    relationship_id: str | None = Field(default=None, max_length=100)
    conversation_summary: str | None = Field(default=None, max_length=4000)
    relationship_facts: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    user_preferences: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


def model_payload(system: str, user: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system + "\n请只输出一个 JSON 对象。"},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }


def call_json(system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    if not (MODEL_BASE_URL and MODEL_API_KEY and MODEL_NAME):
        raise RuntimeError("模型未配置")
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.post(
                f"{MODEL_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {MODEL_API_KEY}"},
                json=model_payload(system, user),
                timeout=httpx.Timeout(25, connect=5),
            )
            response.raise_for_status()
            return schema.model_validate_json(response.json()["choices"][0]["message"]["content"])
        except httpx.HTTPStatusError as error:
            last_error = error
            if error.response.status_code < 500 and error.response.status_code != 429:
                break
        except (httpx.TransportError, ValidationError, KeyError, IndexError, json.JSONDecodeError) as error:
            last_error = error
        if attempt == 0:
            time.sleep(0.2)
    raise RuntimeError("模型响应不可用") from last_error


class CrossEncoderReranker:
    model_name = RERANKER_MODEL

    def __init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            raise RerankerUnavailable(str(exc)) from exc

    def score(self, scene: str, query: str, documents: list[str]) -> list[float]:
        try:
            return [float(value) for value in self._model.predict(
                [[f"{scene}\n{query}", document] for document in documents]
            )]
        except Exception as exc:
            raise RerankerUnavailable(str(exc)) from exc


class UnavailableReranker:
    model_name = RERANKER_MODEL

    def score(self, scene: str, query: str, documents: list[str]) -> list[float]:
        raise RerankerUnavailable("reranker unavailable")


class UnavailableVectorRetriever:
    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise VectorUnavailable("embedding model unavailable")


class DemoRetrievalService:
    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        assessment = EvidenceAssessment(
            status="insufficient",
            covered_topics=[],
            missing_topics=kwargs["plan"].required_topics,
            excluded_topic_hits=[],
            conflicting_chunk_ids=[],
            selected_chunk_ids=[],
            fallback_reason="demo_in_memory",
        )
        trace = RetrievalTrace(
            request_id=kwargs["request_id"],
            conversation_id=kwargs["conversation_id"],
            skill_version=kwargs["skill_version"],
            skill_sha256=kwargs["skill_sha256"],
            corpus_version=kwargs["corpus_version"],
            scene_snapshot={},
            retrieval_plan={},
            lexical_candidates=[],
            vector_candidates=[],
            fused_candidates=[],
            reranked_candidates=[],
            selected_chunks=[],
            evidence_assessment=assessment.model_dump(),
            fallback_reason="demo_in_memory",
            latency_ms={},
        )
        return RetrievalResult([], assessment, trace)


class UnavailableProductionRetrievalService:
    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        raise RuntimeError("production retrieval is not ready")


metrics = RetrievalMetrics()
skill_runtime = SkillRuntime(SKILL_DIR)
_readiness_errors: list[str] = []
_postgres_ready = DEMO_MODE
_corpus_ready = DEMO_MODE
try:
    skill_runtime.load()
except Exception as exc:
    _readiness_errors.append(f"skill: {exc}")
if not DEMO_MODE and not (MODEL_BASE_URL and MODEL_API_KEY and MODEL_NAME):
    _readiness_errors.append("model: provider configuration is incomplete")


def _production_dependencies() -> tuple[object, object, ThreadStore]:
    global _corpus_ready, _postgres_ready
    if not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("production requires DATABASE_URL=postgresql://...")
    import psycopg

    database_url = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    connection = psycopg.connect(database_url)
    _postgres_ready = True
    repository = PostgresKnowledgeRepository(connection)
    expected = CorpusVersion(
        CORPUS_VERSION,
        EMBEDDING_MODEL,
        int(os.getenv("EMBEDDING_DIMENSION", "512")),
        os.getenv("EMBEDDING_NORMALIZATION", "l2"),
        os.getenv("TOKENIZER_VERSION", "jieba-0.42"),
        os.getenv("CHUNKER_VERSION", "heading-semantic-v1"),
    )
    corpus = repository.readiness(expected, allow_lexical_only=ALLOW_LEXICAL_ONLY)
    if not corpus.ready:
        metrics.observe("corpus_version_mismatch_count", 1)
        raise RuntimeError(f"corpus: {corpus.reason}")
    _corpus_ready = True
    try:
        from sentence_transformers import SentenceTransformer

        embedding = SentenceTransformer(EMBEDDING_MODEL)
        vector: object = VectorRetriever(
            repository,
            lambda text: embedding.encode(text, normalize_embeddings=True).tolist(),
            EMBEDDING_MODEL,
        )
    except Exception:
        if not ALLOW_LEXICAL_ONLY:
            raise
        vector = UnavailableVectorRetriever()
    try:
        reranker: object = CrossEncoderReranker()
    except RerankerUnavailable:
        reranker = UnavailableReranker()
    retrieval = RetrievalService(
        LexicalRetriever(connection),
        vector,
        reranker,
        trace_writer=repository,
        metrics=metrics,
        context_loader=repository,
    )
    from langgraph.checkpoint.postgres import PostgresSaver

    global postgres_context
    postgres_context = PostgresSaver.from_conn_string(database_url)
    checkpointer = postgres_context.__enter__()
    checkpointer.setup()
    return retrieval, checkpointer, ThreadStore(database_url, DATA_DIR)


if DEMO_MODE:
    retrieval_service: object = DemoRetrievalService()
    checkpointer: object = SqliteSaver(
        sqlite3.connect(DATA_DIR / "checkpoints.sqlite", check_same_thread=False)
    )
    checkpointer.setup()
    thread_store = ThreadStore("", DATA_DIR)
else:
    try:
        retrieval_service, checkpointer, thread_store = _production_dependencies()
    except Exception as exc:
        _readiness_errors.append(str(exc))
        retrieval_service = UnavailableProductionRetrievalService()
        checkpointer = InMemorySaver()
        thread_store = ThreadStore("", DATA_DIR)

nodes = AssistantNodes(
    skill_runtime=skill_runtime,
    retrieval_service=retrieval_service,
    call_json=call_json,
    metrics=metrics,
    corpus_version=CORPUS_VERSION,
    embedding_model=None if DEMO_MODE else EMBEDDING_MODEL,
    trace_writer=None if DEMO_MODE else getattr(retrieval_service, "trace_writer", None),
    demo_mode=DEMO_MODE,
)
graph = build_assistant_graph(nodes, checkpointer)
graph_lock = threading.RLock()

app = FastAPI(title="CrushPilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    if _readiness_errors:
        response.status_code = 503
    return {
        "ready": not _readiness_errors,
        "mode": "demo" if DEMO_MODE else "production",
        "skill": skill_runtime.readiness().ready,
        "corpus": _corpus_ready,
        "postgres": _postgres_ready,
        "errors": _readiness_errors,
    }


@app.get("/metrics")
def retrieval_metrics() -> dict[str, dict[str, float]]:
    return metrics.snapshot()


def device_id(x_device_id: str = Header(alias="X-Device-Id")) -> str:
    try:
        return str(UUID(x_device_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="无效设备标识") from error


def owned_state(thread_id: str, owner_id: str) -> Any:
    if not thread_store.owns(thread_id, owner_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    with graph_lock:
        return graph.get_state({"configurable": {"thread_id": thread_id}})


@app.post("/api/v1/chat")
def chat(request: ChatRequest, owner_id: str = Depends(device_id)) -> StreamingResponse:
    if _readiness_errors:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    if not thread_store.claim(request.thread_id, owner_id, request.message):
        raise HTTPException(status_code=404, detail="会话不存在")

    def events():
        yield "event: start\ndata: {}\n\n"
        try:
            with graph_lock:
                result = graph.invoke(
                    {
                        "user_message": request.message,
                        "device_id": owner_id,
                        "conversation_id": request.thread_id,
                        "relationship_id": request.relationship_id,
                        "conversation_summary": request.conversation_summary,
                        "relationship_facts": request.relationship_facts,
                        "user_preferences": request.user_preferences,
                        "messages": [{"role": "user", "content": request.message}],
                    },
                    {"configurable": {"thread_id": request.thread_id}},
                )
            yield f"event: complete\ndata: {json.dumps(result['final_response'], ensure_ascii=False)}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'message': '服务暂时不可用，请稍后重试。'}, ensure_ascii=False)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/threads")
def list_threads(owner_id: str = Depends(device_id)) -> list[dict[str, str]]:
    return [
        {"thread_id": thread_id, "title": title}
        for thread_id, title in thread_store.list_for_owner(owner_id)
    ]


@app.get("/api/v1/threads/{thread_id}")
def get_thread(thread_id: str, owner_id: str = Depends(device_id)) -> dict[str, Any]:
    state = owned_state(thread_id, owner_id)
    return {"thread_id": thread_id, "messages": state.values.get("messages", [])}


@app.delete("/api/v1/threads/{thread_id}")
def delete_thread(thread_id: str, owner_id: str = Depends(device_id)) -> dict[str, bool]:
    if not thread_store.delete(thread_id, owner_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    with graph_lock:
        checkpointer.delete_thread(thread_id)
    return {"deleted": True}
