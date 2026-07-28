from __future__ import annotations


class CorpusPublisher:
    def __init__(self, connection: object):
        self.connection = connection

    def publish(self, version: str) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("LOCK TABLE knowledge_corpus_versions IN EXCLUSIVE MODE")
                self._validate(cursor, version, "staging")
                cursor.execute(
                    "UPDATE knowledge_corpus_versions SET status = 'retired' WHERE status = 'published'"
                )
                cursor.execute(
                    """
                    UPDATE knowledge_corpus_versions
                    SET status = 'published', published_at = now()
                    WHERE version = %s
                    """,
                    (version,),
                )
                self._prune(cursor)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def rollback(self, version: str) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("LOCK TABLE knowledge_corpus_versions IN EXCLUSIVE MODE")
                self._validate(cursor, version, "retired")
                cursor.execute(
                    "UPDATE knowledge_corpus_versions SET status = 'retired' WHERE status = 'published'"
                )
                cursor.execute(
                    """
                    UPDATE knowledge_corpus_versions
                    SET status = 'published', published_at = now()
                    WHERE version = %s
                    """,
                    (version,),
                )
                self._prune(cursor)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _validate(cursor: object, version: str, status: str) -> None:
        cursor.execute(
            """
            SELECT
                (SELECT count(*) FROM knowledge_chunks
                 WHERE corpus_version = %s AND review_status = 'approved'
                   AND usage_scope = 'online_eligible'),
                (SELECT count(*) FROM knowledge_embeddings e
                 JOIN knowledge_chunks c
                   ON c.id = e.chunk_id
                  AND c.corpus_version = e.corpus_version
                 WHERE e.corpus_version = %s
                   AND c.review_status = 'approved'
                   AND c.usage_scope = 'online_eligible'
                   AND e.embedding_model = v.embedding_model
                   AND e.embedding_dimension = v.embedding_dimension)
            FROM knowledge_corpus_versions v
            WHERE version = %s AND status = %s
            FOR UPDATE
            """,
            (version, version, version, status),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"{status} corpus version not found")
        if row[0] < 1 or row[1] != row[0]:
            raise ValueError("every approved chunk must have a matching embedding")

    @staticmethod
    def _prune(cursor: object) -> None:
        cursor.execute(
            """
            DELETE FROM knowledge_corpus_versions
            WHERE status = 'retired'
              AND version NOT IN (
                  SELECT version FROM knowledge_corpus_versions
                  WHERE status = 'retired'
                  ORDER BY published_at DESC NULLS LAST, created_at DESC
                  LIMIT 1
              )
            """
        )
