from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.assistant.graph import build_assistant_graph
from app.agents.assistant.nodes import AssistantNodes
from app.core.config import Settings
from app.infrastructure.model_client import OpenAICompatibleJsonClient
from app.knowledge.domain.models import EvidenceAssessment, RetrievalTrace
from app.knowledge.governance.corpus_version import CorpusVersion
from app.knowledge.observability.metrics import RetrievalMetrics
from app.knowledge.repositories.postgres import PostgresKnowledgeRepository
from app.knowledge.retrieval.lexical_retriever import LexicalRetriever
from app.knowledge.retrieval.local_json import LocalJsonRetrievalService
from app.knowledge.retrieval.reranker import RerankerUnavailable
from app.knowledge.retrieval.service import RetrievalResult, RetrievalService
from app.knowledge.retrieval.vector_retriever import VectorRetriever, VectorUnavailable
from app.knowledge.skill.runtime import SkillRuntime
from app.services.chat_service import ChatService
from app.thread_store import ThreadStore


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except Exception as exc:
            raise RerankerUnavailable(str(exc)) from exc

    def score(self, scene: str, query: str, documents: list[str]) -> list[float]:
        try:
            return [
                float(value)
                for value in self._model.predict(
                    [[f"{scene}\n{query}", document] for document in documents]
                )
            ]
        except Exception as exc:
            raise RerankerUnavailable(str(exc)) from exc


class UnavailableReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

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


class AppRuntime:
    """Owns startup resources while preserving degraded readiness semantics."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._started = False
        self._shutdown = False
        self._closers: list[Callable[[], None]] = []
        self.readiness_errors: list[str] = []
        self.postgres_ready = settings.demo_mode
        self.corpus_ready = settings.demo_mode or settings.local_sqlite_mode
        self.metrics = RetrievalMetrics()
        self.graph_lock = threading.RLock()

    def start(self) -> "AppRuntime":
        with self._lock:
            if self._shutdown:
                raise RuntimeError("runtime is shutting down")
            if self._started:
                return self
            self.readiness_errors.clear()
            self.postgres_ready = self.settings.demo_mode
            self.corpus_ready = self.settings.demo_mode or self.settings.local_sqlite_mode
            self.metrics = RetrievalMetrics()
            try:
                self.settings.data_dir.mkdir(parents=True, exist_ok=True)
                self.skill_runtime = SkillRuntime(self.settings.skill_dir)
                try:
                    self.skill_runtime.load()
                except Exception as exc:
                    self.readiness_errors.append(f"skill: {exc}")
                self.model_client = OpenAICompatibleJsonClient(self.settings)
                if not self.settings.demo_mode and not self._model_configured():
                    self.readiness_errors.append("model: provider configuration is incomplete")
                self._assemble_dependencies()
                if (
                    not self.settings.demo_mode
                    and not self.settings.local_sqlite_mode
                    and not self.readiness_errors
                ):
                    try:
                        self.model_client.probe()
                    except Exception:
                        self.readiness_errors.append("model: unavailable")
                self.nodes = AssistantNodes(
                    skill_runtime=self.skill_runtime,
                    retrieval_service=self.retrieval_service,
                    call_json=self.model_client.call_json,
                    metrics=self.metrics,
                    corpus_version=self.settings.corpus_version,
                    embedding_model=(
                        None
                        if self.settings.demo_mode or self.settings.local_sqlite_mode
                        else self.settings.embedding_model
                    ),
                    trace_writer=(
                        None
                        if self.settings.demo_mode or self.settings.local_sqlite_mode
                        else getattr(self.retrieval_service, "trace_writer", None)
                    ),
                    demo_mode=self.settings.demo_mode,
                )
                self.graph = build_assistant_graph(self.nodes, self.checkpointer)
                self.chat_service = ChatService(
                    graph=self.graph,
                    graph_lock=self.graph_lock,
                    checkpointer=self.checkpointer,
                    thread_store=self.thread_store,
                    retrieval_service=self.retrieval_service,
                    local_sqlite_mode=self.settings.local_sqlite_mode,
                    readiness_errors=self.readiness_errors,
                )
                self._finish_pending_deletions()
                self._started = True
                return self
            except Exception:
                self._close_resources()
                raise

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._close_locked()

    def _close_locked(self) -> None:
        if hasattr(self, "chat_service"):
            self.chat_service.begin_draining()
        with self.graph_lock:
            self._close_resources()
        self._started = False

    def _close_resources(self) -> None:
        # 依赖通常按“连接 -> 服务 -> 运行时”装配，逆序释放可避免上层清理时底层已关闭。
        for closer in reversed(self._closers):
            try:
                closer()
            except Exception:
                pass
        self._closers.clear()

    def _finish_pending_deletions(self) -> None:
        # 线程元数据与 LangGraph checkpoint 不在同一事务内。若上次已删 checkpoint
        # 却未删元数据，tombstone 会在下一次启动时继续收敛。
        for thread_id, owner_id in self.thread_store.pending_deletions():
            try:
                self.checkpointer.delete_thread(thread_id)
                self.thread_store.finalize_delete(thread_id, owner_id)
            except Exception:
                pass

    def readiness(self) -> dict[str, object]:
        # readiness 允许惰性启动，便于容器探针得到当前依赖错误，而非直接崩溃。
        self.start()
        return {
            "ready": not self.readiness_errors,
            "mode": (
                "demo"
                if self.settings.demo_mode
                else "local_sqlite"
                if self.settings.local_sqlite_mode
                else "production"
            ),
            "skill": self.skill_runtime.readiness().ready,
            "corpus": self.corpus_ready,
            "postgres": self.postgres_ready,
            "errors": self.readiness_errors,
        }

    def _model_configured(self) -> bool:
        return bool(
            self.settings.model_base_url
            and self.settings.model_api_key
            and self.settings.model_name
        )

    def _assemble_dependencies(self) -> None:
        if self.settings.demo_mode or self.settings.local_sqlite_mode:
            self._assemble_local_dependencies()
            return
        try:
            self._assemble_production_dependencies()
        except Exception as exc:
            self._close_resources()
            self.readiness_errors.append(str(exc))
            self.retrieval_service = UnavailableProductionRetrievalService()
            self.checkpointer = InMemorySaver()
            self.thread_store = ThreadStore("", self.settings.data_dir)
            self._closers.append(self.thread_store.close)

    def _assemble_local_dependencies(self) -> None:
        self.retrieval_service = (
            LocalJsonRetrievalService(self.settings.data_dir / "knowledge_vectors.json")
            if self.settings.local_sqlite_mode
            else DemoRetrievalService()
        )
        connection = sqlite3.connect(
            self.settings.data_dir / "checkpoints.sqlite", check_same_thread=False
        )
        self._closers.append(connection.close)
        self.checkpointer = SqliteSaver(connection)
        self.checkpointer.setup()
        self.thread_store = ThreadStore("", self.settings.data_dir)
        self._closers.append(self.thread_store.close)

    def _assemble_production_dependencies(self) -> None:
        if not self.settings.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise RuntimeError("production requires DATABASE_URL=postgresql://...")
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        database_url = self.settings.database_url.replace(
            "postgresql+psycopg://", "postgresql://", 1
        )
        connection = psycopg.connect(database_url)
        self._closers.append(connection.close)
        context: Any | None = None
        try:
            self.postgres_ready = True
            repository = PostgresKnowledgeRepository(connection)
            expected = CorpusVersion(
                self.settings.corpus_version,
                self.settings.embedding_model,
                self.settings.embedding_dimension,
                self.settings.embedding_normalization,
                self.settings.tokenizer_version,
                self.settings.chunker_version,
            )
            corpus = repository.readiness(
                expected, allow_lexical_only=self.settings.allow_lexical_only
            )
            if not corpus.ready:
                self.metrics.observe("corpus_version_mismatch_count", 1)
                raise RuntimeError(f"corpus: {corpus.reason}")
            self.corpus_ready = True
            try:
                from sentence_transformers import SentenceTransformer

                embedding = SentenceTransformer(self.settings.embedding_model)
                vector: object = VectorRetriever(
                    repository,
                    lambda text: embedding.encode(
                        text, normalize_embeddings=True
                    ).tolist(),
                    self.settings.embedding_model,
                )
            except Exception:
                if not self.settings.allow_lexical_only:
                    raise
                vector = UnavailableVectorRetriever()
            try:
                reranker: object = CrossEncoderReranker(self.settings.reranker_model)
            except RerankerUnavailable:
                reranker = UnavailableReranker(self.settings.reranker_model)
            self.retrieval_service = RetrievalService(
                LexicalRetriever(connection),
                vector,
                reranker,
                trace_writer=repository,
                metrics=self.metrics,
                context_loader=repository,
                trace_debug=self.settings.trace_debug,
            )
            context = PostgresSaver.from_conn_string(database_url)
            self.checkpointer = context.__enter__()
            self._closers.append(lambda: context.__exit__(None, None, None))
            self.checkpointer.setup()
            self.thread_store = ThreadStore(database_url, self.settings.data_dir)
            self._closers.append(self.thread_store.close)
        except Exception:
            raise
