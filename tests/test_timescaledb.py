import os
from datetime import UTC, datetime, timedelta

import psycopg2
import pytest

# Use environment variable or fallback to default
DATABASE_URL = os.getenv("DATABASE_URL_SYNC", "postgresql://admin:29a47839acf362c9ebb5679a@postgres:5432/bsopt")


def is_db_available():
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=1)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def db_conn():
    if not is_db_available():
        pytest.skip("TimescaleDB not available at " + DATABASE_URL)
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    yield conn
    conn.close()


def test_hypertables_exist(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
        hypertables = [row[0] for row in cur.fetchall()]
        assert "options_prices" in hypertables
        assert "model_predictions" in hypertables
        assert "market_ticks" in hypertables


def test_continuous_aggregates_exist(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT view_name FROM timescaledb_information.continuous_aggregates;")
        views = [row[0] for row in cur.fetchall()]
        # Updated to match revamped CAGG names
        assert "minute_stats_cagg" in views
        assert "hourly_stats_chained_cagg" in views
        assert "daily_stats_chained_cagg" in views
        assert "greeks_drift_cagg" in views


def test_insert_and_aggregate(db_conn):
    with db_conn.cursor() as cur:
        # Clear existing data for test symbol
        cur.execute("DELETE FROM options_prices WHERE symbol = 'TEST_AAPL';")
        
        now = datetime.now(UTC)
        # Insert test data into the SAME minute bucket to ensure aggregation works as expected
        bucket_time = now.replace(second=0, microsecond=0) - timedelta(minutes=5)
        
        cur.execute("""
            INSERT INTO options_prices (time, symbol, strike, expiry, option_type, last, volume, implied_volatility, delta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (bucket_time + timedelta(seconds=1), 'TEST_AAPL', 150.00, '2026-06-19', 'call', 10.50, 100, 0.25, 0.55))
        
        cur.execute("""
            INSERT INTO options_prices (time, symbol, strike, expiry, option_type, last, volume, implied_volatility, delta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (bucket_time + timedelta(seconds=2), 'TEST_AAPL', 150.00, '2026-06-19', 'call', 11.00, 200, 0.26, 0.56))

        # Manually refresh the aggregate
        cur.execute("CALL refresh_continuous_aggregate('minute_stats_cagg', NULL, NULL);")
        
        # Verify aggregation
        cur.execute("SELECT symbol, avg_price, volume FROM minute_stats_cagg WHERE symbol = 'TEST_AAPL';")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 'TEST_AAPL'
        assert float(row[1]) == 10.75
        assert row[2] == 300


def test_model_predictions_aggregate(db_conn):
    with db_conn.cursor() as cur:
        # Clear existing
        cur.execute("DELETE FROM model_predictions;")
        
        now = datetime.now(UTC)
        # Insert predictions
        cur.execute("""
            INSERT INTO model_predictions (timestamp, symbol, predicted_price, actual_price, model_id, input_features)
            VALUES (%s, %s, %s, %s, NULL, '{}')
        """, (now - timedelta(hours=2), 'AAPL', 150.00, 151.00))
        
        cur.execute("""
            INSERT INTO model_predictions (timestamp, symbol, predicted_price, actual_price, model_id, input_features)
            VALUES (%s, %s, %s, %s, NULL, '{}')
        """, (now - timedelta(hours=1), 'AAPL', 155.00, 154.00))

        # Verify drift metrics view (which replaced the old model_daily_performance cagg)
        cur.execute("SELECT * FROM model_drift_metrics_mv;")
        # Since model_id is NULL, they might not show up if the view groups by model_id
        # Let's check the view definition or insert a dummy model
        
        cur.execute("""
            INSERT INTO ml_models (id, name, algorithm, version, model_artifact_url) 
            VALUES (%s, %s, %s, %s, %s) 
            ON CONFLICT (name, version) DO UPDATE SET id = EXCLUDED.id, model_artifact_url = EXCLUDED.model_artifact_url
        """, ('00000000-0000-0000-0000-000000000001', 'test_model', 'xgboost', 1, 'http://test'))
        
        cur.execute("""
            INSERT INTO model_predictions (timestamp, symbol, predicted_price, actual_price, model_id, input_features)
            VALUES (%s, %s, %s, %s, %s, '{}')
        """, (now - timedelta(minutes=30), 'AAPL', 100.00, 102.00, '00000000-0000-0000-0000-000000000001'))

        cur.execute("REFRESH MATERIALIZED VIEW model_drift_metrics_mv;")
        
        cur.execute("SELECT mae FROM model_drift_metrics_mv WHERE model_id = '00000000-0000-0000-0000-000000000001';")
        row = cur.fetchone()
        assert row is not None
        assert float(row[0]) == 2.0  # abs(100 - 102)
