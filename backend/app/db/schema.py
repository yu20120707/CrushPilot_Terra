from __future__ import annotations


SCHEMA_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS knowledge_corpus_versions (
        version varchar PRIMARY KEY,
        status varchar NOT NULL CHECK (status IN ('staging', 'published', 'retired')),
        embedding_model varchar NOT NULL,
        embedding_dimension integer NOT NULL CHECK (embedding_dimension > 0),
        embedding_normalization varchar NOT NULL,
        tokenizer_version varchar NOT NULL,
        chunker_version varchar NOT NULL,
        published_at timestamptz NULL,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_single_published
    ON knowledge_corpus_versions(status) WHERE status = 'published'
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_documents (
        id uuid PRIMARY KEY,
        source_path text NOT NULL,
        title text NOT NULL,
        source_sha256 varchar NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        corpus_version varchar NOT NULL REFERENCES knowledge_corpus_versions(version) ON DELETE CASCADE,
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE(source_path, corpus_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_sections (
        id uuid PRIMARY KEY,
        document_id uuid NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
        parent_section_id uuid NULL REFERENCES knowledge_sections(id) ON DELETE CASCADE,
        title text NOT NULL,
        heading_path jsonb NOT NULL,
        summary text NULL,
        ordinal integer NOT NULL,
        corpus_version varchar NOT NULL REFERENCES knowledge_corpus_versions(version) ON DELETE CASCADE,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE knowledge_sections ADD COLUMN IF NOT EXISTS summary text NULL",
    """
    CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id uuid PRIMARY KEY,
        document_id uuid NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
        parent_section_id uuid NULL REFERENCES knowledge_sections(id) ON DELETE SET NULL,
        title text NOT NULL,
        heading_path jsonb NOT NULL,
        content text NOT NULL,
        metadata jsonb NOT NULL,
        search_tokens text NOT NULL,
        search_vector tsvector NOT NULL,
        review_status varchar NOT NULL,
        source_sha256 varchar NOT NULL,
        corpus_version varchar NOT NULL REFERENCES knowledge_corpus_versions(version) ON DELETE CASCADE,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_knowledge_chunks_id_version UNIQUE(id, corpus_version)
    )
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'uq_knowledge_chunks_id_version'
              AND conrelid = 'knowledge_chunks'::regclass
        ) THEN
            ALTER TABLE knowledge_chunks
            ADD CONSTRAINT uq_knowledge_chunks_id_version UNIQUE(id, corpus_version);
        END IF;
    END $$
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON knowledge_chunks USING GIN(search_vector)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON knowledge_chunks USING GIN(metadata)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_corpus_status ON knowledge_chunks(corpus_version, review_status)",
    """
    CREATE TABLE IF NOT EXISTS knowledge_embeddings (
        chunk_id uuid NOT NULL,
        embedding_model varchar NOT NULL,
        embedding_dimension integer NOT NULL CHECK (embedding_dimension > 0),
        embedding vector NOT NULL,
        corpus_version varchar NOT NULL REFERENCES knowledge_corpus_versions(version) ON DELETE CASCADE,
        PRIMARY KEY(chunk_id, embedding_model, corpus_version),
        CONSTRAINT fk_embeddings_chunk_version FOREIGN KEY(chunk_id, corpus_version)
            REFERENCES knowledge_chunks(id, corpus_version) ON DELETE CASCADE
    )
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'fk_embeddings_chunk_version'
              AND conrelid = 'knowledge_embeddings'::regclass
        ) THEN
            ALTER TABLE knowledge_embeddings
            ADD CONSTRAINT fk_embeddings_chunk_version
            FOREIGN KEY(chunk_id, corpus_version)
            REFERENCES knowledge_chunks(id, corpus_version) ON DELETE CASCADE;
        END IF;
    END $$
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_traces (
        request_id varchar PRIMARY KEY,
        conversation_id varchar NOT NULL,
        skill_version varchar NOT NULL,
        skill_sha256 varchar NOT NULL,
        corpus_version varchar NOT NULL,
        embedding_model varchar NULL,
        reranker_model varchar NULL,
        scene_snapshot jsonb NOT NULL,
        retrieval_plan jsonb NOT NULL,
        evidence_assessment jsonb NOT NULL,
        fallback_reason text NULL,
        latency_ms jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_trace_candidates (
        request_id varchar NOT NULL REFERENCES retrieval_traces(request_id) ON DELETE CASCADE,
        stage varchar NOT NULL,
        chunk_id uuid NULL,
        rank integer NOT NULL,
        score double precision NULL,
        details jsonb NOT NULL DEFAULT '{}'::jsonb,
        PRIMARY KEY(request_id, stage, rank)
    )
    """,
)


def apply_schema(connection: object) -> None:
    with connection.cursor() as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
    connection.commit()
