from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_VERSION = "2026.08.1"
DEFAULT_NEW_KB_ROOT = ROOT / "knowledge" / "pua-knowledge-sharing"
sys.path.insert(0, str(ROOT / "backend"))

from app.db.schema import apply_schema
from app.knowledge.ingestion import build_corpus, write_build_artifacts
from app.knowledge.governance.corpus_version import CorpusVersion
from app.knowledge.domain.models import topic_search_text
from app.knowledge.repositories.postgres import PostgresKnowledgeRepository
from app.knowledge.repositories.publisher import CorpusPublisher
from app.knowledge.retrieval.tokenizer import tokenize


def search_tokens(build) -> dict[str, dict[str, str]]:
    result = {}
    for record in build.chunks:
        chunk = record.chunk
        result[chunk.chunk_id] = {
            "A": " ".join(tokenize(" ".join((chunk.title, *chunk.topics, *chunk.action_labels)))),
            "B": " ".join(
                tokenize(
                    " ".join(
                        (
                            *chunk.heading_path,
                            *chunk.applicable_when,
                            *chunk.not_applicable_when,
                        )
                    )
                )
            ),
            "C": " ".join(tokenize(chunk.content)),
            "D": " ".join(tokenize(chunk.source_path)),
        }
    return result


def embedding_text(chunk) -> str:
    return "\n".join(
        (
            chunk.title,
            " / ".join(chunk.heading_path),
            topic_search_text(chunk.topics),
            chunk.content,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--version", default=DEFAULT_CORPUS_VERSION)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument(
        "--new-kb-root", type=Path, default=DEFAULT_NEW_KB_ROOT,
        help="Optional Markdown root ingested as source_collection=new_kb; all files remain auditable.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "trellis" / "retrieval-system-refactor",
    )
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")

    from sentence_transformers import SentenceTransformer
    import psycopg

    model = SentenceTransformer(args.embedding_model)
    dimension = int(model.get_sentence_embedding_dimension())
    build = build_corpus(
        args.version,
        embedding_model=args.embedding_model,
        embedding_dimension=dimension,
        new_kb_root=args.new_kb_root,
    )
    write_build_artifacts(build, args.artifacts)
    vectors = model.encode(
        [embedding_text(record.chunk) for record in build.chunks],
        normalize_embeddings=True,
    )
    embeddings = {
        record.chunk.chunk_id: vector.tolist()
        for record, vector in zip(build.chunks, vectors, strict=True)
    }
    with psycopg.connect(args.database_url) as connection:
        apply_schema(connection)
        repository = PostgresKnowledgeRepository(connection)
        expected = CorpusVersion(
            args.version,
            args.embedding_model,
            build.report.embedding_dimension,
            build.report.embedding_normalization,
            build.report.tokenizer_version,
            build.report.chunker_version,
        )
        if repository.readiness(expected).ready:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_collection, count(*)
                    FROM knowledge_chunks
                    WHERE corpus_version = %s
                    GROUP BY source_collection
                    """,
                    (args.version,),
                )
                collections = dict(cursor.fetchall())
            required_collections = {"original", "new_kb"} if args.new_kb_root else {"original"}
            if required_collections <= set(collections):
                print(f"published_corpus={args.version} already_ready=true collections={collections}")
                return
            raise SystemExit(
                f"published_corpus={args.version} is missing required collections: "
                f"expected={sorted(required_collections)} actual={sorted(collections)}"
            )
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM knowledge_corpus_versions "
                "WHERE version = %s AND status = 'staging'",
                (args.version,),
            )
        connection.commit()
        repository.write_staging_build(build, search_tokens(build))
        repository.write_embeddings(args.version, args.embedding_model, embeddings)
        repository.link_supersedes_from_published(args.version)
        CorpusPublisher(connection).publish(args.version)
    print(
        f"published_corpus={args.version} chunks={len(build.chunks)} "
        f"collections={build.report.collection_chunk_counts} "
        f"usage_scopes={build.report.usage_scope_counts}"
    )


if __name__ == "__main__":
    main()
