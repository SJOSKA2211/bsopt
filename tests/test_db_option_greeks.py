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

def test_option_greeks_table_exists(db_engine):
    """Test that option_greeks table exists."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT to_regclass('option_greeks');"))
        assert result.scalar() is not None, "option_greeks table does not exist"

def test_option_greeks_is_hypertable(db_engine):
    """Test that option_greeks is a hypertable."""
    with db_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'option_greeks';"
        ))
        assert result.fetchone() is not None, "option_greeks is not a hypertable"

def test_insert_and_query_greeks(db_engine):
    """Test inserting and retrieving option greeks."""
    with db_engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE option_greeks;"))
            conn.commit()
        except Exception:
            conn.rollback()
            
        timestamp = datetime.now()
        contract_id = "AAPL-20260116-C-150"
        delta = 0.55
        gamma = 0.02
        theta = -0.05
        vega = 0.12
        rho = 0.01
        implied_vol = 0.25
        
        conn.execute(text("""
            INSERT INTO option_greeks (time, contract_id, delta, gamma, theta, vega, rho, implied_vol)
            VALUES (:time, :contract_id, :delta, :gamma, :theta, :vega, :rho, :implied_vol)
        """), {
            "time": timestamp,
            "contract_id": contract_id,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "implied_vol": implied_vol
        })
        conn.commit()
        
        # Query
        result = conn.execute(text("SELECT * FROM option_greeks WHERE contract_id = :contract_id"), {"contract_id": contract_id})
        row = result.fetchone()
        assert row is not None
        assert float(row.delta) == delta
