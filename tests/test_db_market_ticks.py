import os
from datetime import datetime

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

def test_market_ticks_table_exists(db_engine):
    """Test that market_ticks table exists."""
    with db_engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("SELECT to_regclass('market_ticks');"))
        assert result.scalar() is not None, "market_ticks table does not exist"

def test_market_ticks_is_hypertable(db_engine):
    """Test that market_ticks is a hypertable."""
    with db_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_ticks';"
        ))
        assert result.fetchone() is not None, "market_ticks is not a hypertable"

def test_insert_and_query_tick(db_engine):
    """Test inserting and retrieving a tick."""
    with db_engine.connect() as conn:
        try:
            # Clean up (might fail if table doesn't exist, but that's fine for Red phase)
            conn.execute(text("TRUNCATE market_ticks;"))
            conn.commit()
        except Exception:
            conn.rollback()
        
        # Insert
        timestamp = datetime.now()
        symbol = "TEST-USD"
        price = 100.50
        volume = 10.0
        
        conn.execute(text(
            "INSERT INTO market_ticks (time, symbol, price, volume) VALUES (:time, :symbol, :price, :volume)"
        ), {"time": timestamp, "symbol": symbol, "price": price, "volume": volume})
        conn.commit()
        
        # Query
        result = conn.execute(text("SELECT * FROM market_ticks WHERE symbol = :symbol"), {"symbol": symbol})
        row = result.fetchone()
        assert row is not None
        assert row.price == price
