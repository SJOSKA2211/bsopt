import pytest
from sqlalchemy import create_engine, inspect

from src.database.models import MarketTick


@pytest.fixture
def test_engine():
    """Create a memory SQLite engine for schema validation."""
    engine = create_engine("sqlite:///:memory:")
    # Only create the table we care about to avoid JSONB/other Postgres specific issues
    MarketTick.__table__.create(engine)
    return engine

def test_market_data_symbol_tagging(test_engine):
    """Verify that market_data table has consistent symbol tagging for TFT group_ids."""
    inspector = inspect(test_engine)
    columns = inspector.get_columns("market_ticks")
    column_names = [c["name"] for c in columns]
    
    assert "symbol" in column_names
    assert "time" in column_names
    assert "price" in column_names
    
    # Check if symbol is indexed or part of a unique constraint
    indexes = inspector.get_indexes("market_ticks")
    # In SQLite, we check if symbol is in any index
    has_symbol_index = any("symbol" in idx["column_names"] for idx in indexes)
    assert has_symbol_index, "symbol column should be indexed for TFT grouping performance"

def test_timescaledb_hypertable_status_mocked(test_engine):
    """
    Simulate hypertable check. 
    In unit tests, we verify the model exists. 
    Actual hypertable conversion is handled via SQL migration.
    """
    inspector = inspect(test_engine)
    assert "market_ticks" in inspector.get_table_names()