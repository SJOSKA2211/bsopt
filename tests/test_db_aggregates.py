import pytest
import os
import sqlalchemy
from sqlalchemy import text

# Skip if no DB connection
DATABASE_URL = os.getenv("DATABASE_URL")

@pytest.fixture(scope="module")
def db_engine():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    return sqlalchemy.create_engine(DATABASE_URL)

def test_candles_view_exists(db_engine):
    """Test that market_candles_1m view exists."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT to_regclass('market_candles_1m');"))
        assert result.scalar() is not None, "market_candles_1m view does not exist"

def test_compression_policy_exists(db_engine):
    """Test that compression policy is enabled for market_ticks."""
    with db_engine.connect() as conn:
        # Check compression settings on hypertable
        result = conn.execute(text(
            "SELECT * FROM timescaledb_information.compression_settings WHERE hypertable_name = 'market_ticks';"
        ))
        assert result.fetchone() is not None, "Compression not configured for market_ticks"
        
        # Check policy job
        result = conn.execute(text(
            "SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression' AND hypertable_name = 'market_ticks';"
        ))
        assert result.fetchone() is not None, "Compression policy job not found"
