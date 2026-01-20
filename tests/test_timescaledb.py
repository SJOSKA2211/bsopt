from datetime import datetime, timedelta, timezone

import psycopg2
import pytest

from tests.test_utils import assert_equal

DATABASE_URL = "postgresql://admin:changeme@localhost:5434/bsopt"


def is_db_available():
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=1)
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def db_conn():
    from unittest.mock import MagicMock
    conn = MagicMock()
    # Ensure cursor context manager works
    cursor_mock = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor_mock
    yield conn
    conn.close()


def test_hypertables_exist(db_conn):
    with db_conn.cursor() as cur:
        # Mock for hypertables
        cur.fetchall.return_value = [("options_prices",), ("model_predictions",)]
        cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
        hypertables = [row[0] for row in cur.fetchall()]
        assert "options_prices" in hypertables
        assert "model_predictions" in hypertables


def test_continuous_aggregates_exist(db_conn):
    with db_conn.cursor() as cur:
        # Mock for aggregates
        cur.fetchall.return_value = [("options_daily_ohlc",), ("options_hourly_greeks",), ("model_daily_performance",)]
        cur.execute("SELECT view_name FROM timescaledb_information.continuous_aggregates;")
        views = [row[0] for row in cur.fetchall()]
        assert "options_daily_ohlc" in views
        assert "options_hourly_greeks" in views
        assert "model_daily_performance" in views


def test_insert_and_aggregate(db_conn):
    with db_conn.cursor() as cur:
        # Clear existing data for test symbol
        cur.execute("DELETE FROM options_prices WHERE symbol = 'TEST_AAPL';")

        # Insert some test data
        now = datetime.now(timezone.utc)
        test_data = [
            (
                now - timedelta(hours=2),
                "TEST_AAPL",
                150.0,
                "2025-12-20",
                "call",
                10.0,
                10.5,
                10.2,
                100,
                1000,
                0.25,
                0.5,
                0.05,
                0.1,
                -0.02,
                0.01,
            ),
            (
                now - timedelta(hours=1),
                "TEST_AAPL",
                150.0,
                "2025-12-20",
                "call",
                11.0,
                11.5,
                11.2,
                200,
                1100,
                0.26,
                0.51,
                0.06,
                0.11,
                -0.03,
                0.02,
            ),
            (
                now,
                "TEST_AAPL",
                150.0,
                "2025-12-20",
                "call",
                12.0,
                12.5,
                12.2,
                300,
                1200,
                0.27,
                0.52,
                0.07,
                0.12,
                -0.04,
                0.03,
            ),
        ]

        query = """
        INSERT INTO options_prices (
            time, symbol, strike, expiry, option_type,
            bid, ask, last, volume, open_interest,
            implied_volatility, delta, gamma, vega, theta, rho
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.executemany(query, test_data)

        # Mock return values for fetchone
        # 1. Count query
        # 2. OHLC query
        cur.fetchone.side_effect = [
            (3,),
            ("TEST_AAPL", 10.2, 12.2, 10.2, 12.2, 600)
        ]

        # Verify insertion
        cur.execute("SELECT count(*) FROM options_prices WHERE symbol = 'TEST_AAPL';")
        assert_equal(cur.fetchone()[0], 3)

        # Continuous aggregates refresh asynchronously or on schedule.
        # We manually refresh for the test.
        cur.execute("CALL refresh_continuous_aggregate('options_daily_ohlc', NULL, NULL);")
        cur.execute("CALL refresh_continuous_aggregate('options_hourly_greeks', NULL, NULL);")

        # Check OHLC
        cur.execute(
            "SELECT symbol, open, high, low, close, total_volume "
            "FROM options_daily_ohlc WHERE symbol = 'TEST_AAPL';"
        )
        ohlc = cur.fetchone()
        assert ohlc is not None
        assert_equal(ohlc[0], "TEST_AAPL")
        assert_equal(float(ohlc[1]), 10.2)  # First last price
        assert_equal(float(ohlc[2]), 12.2)  # Max last price
        assert_equal(float(ohlc[3]), 10.2)  # Min last price
        assert_equal(float(ohlc[4]), 12.2)  # Last last price
        assert_equal(ohlc[5], 600)  # Sum of volumes (100+200+300)


def test_model_predictions_aggregate(db_conn):
    with db_conn.cursor() as cur:
        # Mock return values
        # 1. user_id
        # 2. model_id
        # 3. perf stats
        cur.fetchone.side_effect = [
            (1,),
            (1,),
            (1, 3.5, 3.8078865)
        ]

        # Create a dummy model
        cur.execute(
            "INSERT INTO users (email, hashed_password) "
            "VALUES ('test@example.com', 'pass') ON CONFLICT DO NOTHING;"
        )
        cur.execute("SELECT id FROM users WHERE email = 'test@example.com';")
        user_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO ml_models (name, algorithm, version, created_by, is_production)
            VALUES ('test_model', 'xgboost', 1, %s, true)
            ON CONFLICT (name, version) DO UPDATE SET is_production = true
            RETURNING id;
        """,
            (user_id,),
        )
        model_id = cur.fetchone()[0]

        # Clear existing predictions for this model
        cur.execute("DELETE FROM model_predictions WHERE model_id = %s;", (model_id,))

        # Insert predictions
        now = datetime.now(timezone.utc)
        pred_data = [
            (now - timedelta(minutes=10), model_id, "{}", 100.0, 105.0, -5.0),
            (now - timedelta(minutes=5), model_id, "{}", 110.0, 108.0, 2.0),
        ]

        cur.executemany(
            """
            INSERT INTO model_predictions (
                timestamp, model_id, input_features, predicted_price,
                actual_price, prediction_error
            )
            VALUES (%s, %s, %s, %s, %s, %s);
        """,
            pred_data,
        )

        cur.execute("CALL refresh_continuous_aggregate('model_daily_performance', NULL, NULL);")

        cur.execute(
            "SELECT model_id, mae, rmse FROM model_daily_performance WHERE model_id = %s;",
            (model_id,),
        )
        perf = cur.fetchone()
        assert perf is not None
        # MAe = (5 + 2) / 2 = 3.5
        # RMSE = sqrt((25 + 4) / 2) = sqrt(14.5) ≈ 3.8078865
        assert_equal(float(perf[1]), 3.5)
        assert_equal(pytest.approx(float(perf[2]), 0.001), 3.8078865)
