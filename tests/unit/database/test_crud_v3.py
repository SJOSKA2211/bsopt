from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.database import crud
from src.database.models import (
    MLModel,
    Order,
    Portfolio,
    Position,
    User,
)

@pytest.mark.asyncio
async def test_crud_basics():
    db = AsyncMock()
    # Mock for get calls
    mock_res = MagicMock()
    db.execute.return_value = mock_res

    # 1. User ops
    mock_user = User(id=uuid4(), email="test@example.com", tier="free")
    mock_res.scalar_one_or_none.return_value = mock_user
    assert await crud.get_user_by_id(db, mock_user.id) == mock_user
    assert await crud.get_user_by_email(db, "test@example.com") == mock_user

    # 2. Portfolio ops
    mock_port = Portfolio(id=uuid4(), user_id=mock_user.id, name="Test")
    mock_res.scalar_one_or_none.return_value = mock_port
    assert await crud.get_portfolio_by_id(db, mock_port.id) == mock_port

    # 3. Position ops
    mock_pos = Position(id=uuid4(), symbol="AAPL", quantity=10)
    mock_res.scalar_one_or_none.return_value = mock_pos
    assert await crud.get_position_by_id(db, mock_pos.id) == mock_pos

    # 4. Order ops
    mock_order = Order(id=uuid4(), symbol="AAPL", side="BUY")
    mock_res.scalar_one_or_none.return_value = mock_order
    assert await crud.get_order_by_id(db, mock_order.id) == mock_order

@pytest.mark.asyncio
async def test_bulk_inserts_all():
    db = AsyncMock()
    mock_conn = AsyncMock()
    mock_raw = AsyncMock()
    # Force fallback to standard execute for all bulk inserts
    driver = MagicMock(spec=[])
    db.connection.return_value = mock_conn
    mock_conn.get_raw_connection.return_value = mock_raw
    mock_raw.driver_connection = driver

    # Market Ticks
    await crud.bulk_insert_market_ticks(db, [{"symbol": "A", "price": 1.0}])
    # Option Prices
    await crud.bulk_insert_option_prices(db, [{"symbol": "O", "price": 2.0}])
    # Audit Logs
    await crud.bulk_insert_audit_logs(db, [{"event_type": "test"}])
    # Request Logs
    await crud.bulk_insert_request_logs(db, [{"path": "/"}])

    assert db.execute.called
    assert db.commit.called

@pytest.mark.asyncio
async def test_model_ops():
    db = AsyncMock()
    model_id = uuid4()
    mock_model = MLModel(id=model_id, name="m1", version=1)
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = mock_model
    db.execute.return_value = mock_res

    # get_latest
    with patch("src.database.crud.get_latest_model_version", AsyncMock(return_value=mock_model)):
        res = await crud.create_model(db, "m1", "algo")
        assert res.version == 2

    # set_prod
    await crud.set_production_model(db, model_id)
    assert db.commit.called

@pytest.mark.asyncio
async def test_mv_queries():
    db = AsyncMock()
    mock_row = MagicMock()
    mock_row._mapping = {
        "total_pnl": 100.0,
        "win_rate": 0.5,
        "hour": datetime.now(UTC),
        "symbol": "A",
        "total_quantity": 10,
    }

    mock_res = MagicMock()
    mock_res.__iter__.return_value = [mock_row]
    mock_res.first.return_value = mock_row
    db.execute.return_value = mock_res

    # Portfolio summary
    res = await crud.get_portfolio_summary(db, uuid4())
    assert len(res) == 1

    # Trading stats (uses .first() and manual mapping)
    # Actually get_user_trading_stats uses total_orders etc
    mock_row._mapping.update(
        {
            "total_orders": 10,
            "filled_orders": 5,
            "cancelled_orders": 0,
            "fill_rate": 0.5,
            "avg_fill_price": 100.0,
        }
    )
    res = await crud.get_user_trading_stats(db, uuid4())
    assert res["total_orders"] == 10

    # IV Surface
    res = await crud.get_iv_surface(db, "AAPL")
    assert len(res) == 1

    # Hourly stats
    res = await crud.get_hourly_market_stats(db, "AAPL")
    assert len(res) == 1

@pytest.mark.asyncio
async def test_closing_and_expiring():
    db = AsyncMock()
    pos_id = uuid4()
    mock_pos = Position(
        id=pos_id, entry_price=Decimal("100"), quantity=Decimal("10"), status="open"
    )

    with patch("src.database.crud.get_position_by_id", AsyncMock(return_value=mock_pos)):
        res = await crud.close_position(db, pos_id, Decimal("110"))
        assert res.status == "closed"
        assert res.realized_pnl == Decimal("100")

    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = [mock_pos]
    db.execute.return_value = mock_res
    res = await crud.get_expiring_positions(db, 5)
    assert len(res) == 1
