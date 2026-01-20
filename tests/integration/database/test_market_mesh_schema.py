import pytest
import psycopg2
import os

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL environment variable not set, skipping integration tests."
)

@pytest.fixture
def db_connection():
    db_url = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/bsopt")
    conn = psycopg2.connect(db_url)
    yield conn
    conn.close()

@pytest.mark.integration
def test_market_data_mesh_table_exists(db_connection):
    cur = db_connection.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'market_data_mesh'
        );
    """)
    assert cur.fetchone()[0] is True

@pytest.mark.integration
def test_market_data_mesh_columns(db_connection):
    cur = db_connection.cursor()
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'market_data_mesh';
    """)
    columns = [row[0] for row in cur.fetchall()]
    assert "symbol" in columns
    assert "market" in columns
    assert "source_type" in columns
    assert "close" in columns
