import os
from datetime import date

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

def test_option_contracts_table_exists(db_engine):
    """Test that option_contracts table exists."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT to_regclass('option_contracts');"))
        assert result.scalar() is not None, "option_contracts table does not exist"

def test_option_contracts_indices(db_engine):
    """Test that optimized indices exist for option_contracts."""
    with db_engine.connect() as conn:
        result = conn.execute(text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'option_contracts' 
            AND indexname IN ('ix_option_contracts_underlying_expiry_strike');
        """))
        assert result.fetchone() is not None, "Optimized composite index is missing"

def test_insert_and_query_option(db_engine):
    """Test inserting and retrieving an option contract."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE option_contracts CASCADE;"))
            conn.commit()
        except Exception:
            conn.rollback()
            
        # Insert a sample contract
        conn.execute(text("""
            INSERT INTO option_contracts (id, underlying, expiry, strike, option_type)
            VALUES (:id, :underlying, :expiry, :strike, :option_type)
        """), {
            "id": "AAPL-20260116-C-150",
            "underlying": "AAPL",
            "expiry": date(2026, 1, 16),
            "strike": 150.00,
            "option_type": "call"
        })
        conn.commit()
        
        # Query
        result = conn.execute(text("""
            SELECT * FROM option_contracts 
            WHERE underlying = 'AAPL' AND expiry = :expiry AND strike = :strike
        """), {"expiry": date(2026, 1, 16), "strike": 150.00})
        row = result.fetchone()
        assert row is not None
        assert row.option_type == "call"
