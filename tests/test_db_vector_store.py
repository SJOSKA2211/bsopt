import os

import pytest
import sqlalchemy
from sqlalchemy import text

# Skip if no DB connection
DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture(scope="module")
def db_engine():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    return sqlalchemy.create_engine(DATABASE_URL)


def test_model_embeddings_table_exists(db_engine):
    """Test that model_embeddings table exists."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT to_regclass('model_embeddings');"))
        assert result.scalar() is not None, "model_embeddings table does not exist"


def test_vector_similarity_search(db_engine):
    """Test L2 distance similarity search using pgvector."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE model_embeddings;"))
            conn.commit()
        except Exception:
            conn.rollback()

        # Insert test vectors (1536 dims)
        vec_a = [1.0] * 1536
        vec_b = [10.0] * 1536
        target = [2.0] * 1536

        conn.execute(
            text(
                """
            INSERT INTO model_embeddings (model_id, version, embedding)
            VALUES ('model_a', 1, :embedding)
        """
            ),
            {"embedding": str(vec_a)},
        )

        conn.execute(
            text(
                """
            INSERT INTO model_embeddings (model_id, version, embedding)
            VALUES ('model_b', 1, :embedding)
        """
            ),
            {"embedding": str(vec_b)},
        )
        conn.commit()

        # Search for neighbor
        result = conn.execute(
            text(
                """
            SELECT model_id FROM model_embeddings
            ORDER BY embedding <-> :target
            LIMIT 1
        """
            ),
            {"target": str(target)},
        )
        row = result.fetchone()
        assert row is not None
        assert row.model_id == "model_a"


def test_hnsw_index_exists(db_engine):
    """Test that HNSW index exists for embeddings."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'model_embeddings' AND indexdef LIKE '%hnsw%';
        """
            )
        )
        assert result.fetchone() is not None, "HNSW index is missing"
