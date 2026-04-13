from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.database import crud


@pytest.mark.asyncio
async def test_bulk_inserts_optimized_paths():
    """
    Test the optimized bulk insert paths (copy_records_to_table).
    """
    db = AsyncMock()
    mock_conn = AsyncMock()
    mock_raw = AsyncMock()

    # Setup driver connection with copy_records_to_table
    driver_conn = MagicMock()
    driver_conn.copy_records_to_table = AsyncMock()

    mock_raw.driver_connection = driver_conn
    mock_conn.get_raw_connection.return_value = mock_raw
    db.connection.return_value = mock_conn

    # 1. Option Prices
    option_prices = [
        {
            "time": datetime.now(UTC),
            "symbol": "AAPL",
            "strike": Decimal("150.0"),
            "expiry": datetime.now(UTC).date(),
            "option_type": "call",
            "bid": Decimal("1.0"),
            "ask": Decimal("1.1"),
            "last": Decimal("1.05"),
            "volume": 100,
            "open_interest": 1000,
            "implied_volatility": Decimal("0.2"),
            "delta": Decimal("0.5"),
            "gamma": Decimal("0.1"),
            "vega": Decimal("0.05"),
            "theta": Decimal("-0.05"),
            "rho": Decimal("0.01"),
        }
    ]

    await crud.bulk_insert_option_prices(db, option_prices)

    assert driver_conn.copy_records_to_table.called
    call_args = driver_conn.copy_records_to_table.call_args
    assert call_args[0][0] == "staging_option_prices"
    records = call_args[1]["records"]
    assert len(records) == 1
    assert len(records[0]) == 16  # 16 columns
    assert records[0][1] == "AAPL"  # symbol is at index 1

    # Reset
    driver_conn.copy_records_to_table.reset_mock()

    # 2. Market Ticks
    ticks = [
        {
            "time": datetime.now(UTC),
            "symbol": "TSLA",
            "price": Decimal("200.0"),
            "volume": 50,
            "side": "buy",
        }
    ]

    await crud.bulk_insert_market_ticks(db, ticks)

    assert driver_conn.copy_records_to_table.called
    call_args = driver_conn.copy_records_to_table.call_args
    assert call_args[0][0] == "staging_market_ticks"
    records = call_args[1]["records"]
    assert len(records) == 1
    assert len(records[0]) == 5  # 5 columns
    assert records[0][1] == "TSLA"

    # Reset
    driver_conn.copy_records_to_table.reset_mock()

    # 3. Audit Logs
    audit_logs = [{"event_type": "login", "user_id": "u1", "details": {"ip": "127.0.0.1"}}]

    await crud.bulk_insert_audit_logs(db, audit_logs)

    assert driver_conn.copy_records_to_table.called
    call_args = driver_conn.copy_records_to_table.call_args
    assert call_args[0][0] == "audit_logs"
    records = call_args[1]["records"]
    assert len(records) == 1
    # Check details serialization
    # Index for details in audit_logs is 8
    # "id", "event_type", "user_id", "user_email", "source_ip", "user_agent", "request_path", "request_method", "details", "created_at"
    assert isinstance(records[0][8], str)  # Should be string (decoded orjson)

    # Reset
    driver_conn.copy_records_to_table.reset_mock()

    # 4. Request Logs
    request_logs = [{"path": "/api/v1/users", "headers": {"User-Agent": "Bolt"}}]

    await crud.bulk_insert_request_logs(db, request_logs)

    assert driver_conn.copy_records_to_table.called
    call_args = driver_conn.copy_records_to_table.call_args
    assert call_args[0][0] == "request_logs"
    records = call_args[1]["records"]
    assert len(records) == 1
    # Check headers serialization
    # Index for headers is 5
    # "id", "request_id", "method", "path", "query_params", "headers", ...
    assert isinstance(records[0][5], str)  # Should be string (decoded orjson)