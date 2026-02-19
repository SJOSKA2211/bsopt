from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import crud
from src.database.models import Position, User


@pytest.fixture
def mock_db():
    session = AsyncMock(spec=AsyncSession)

    # Mock result scalars
    class MockResult:
        def __init__(self, items=None, scalar=None):
            self._items = items or []
            self._scalar = scalar

        def scalar_one_or_none(self):
            return self._scalar

        def scalars(self):
            m = MagicMock()
            m.all.return_value = self._items
            return m

        def first(self):
            return self._items[0] if self._items else None

        @property
        def rowcount(self):
            return 1

    session.mock_result_cls = MockResult
    return session


@pytest.mark.asyncio
async def test_get_user_by_id(mock_db):
    user_id = uuid4()
    mock_user = User(id=user_id, email="test@test.com")

    mock_db.execute.return_value = mock_db.mock_result_cls(scalar=mock_user)

    result = await crud.get_user_by_id(mock_db, user_id)
    assert result == mock_user
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_create_user(mock_db):
    email = "new@test.com"
    password = "password123"
    full_name = "Rick Sanchez"

    # Mock password hashing
    with patch(
        "src.security.password.password_service.hash_password",
        return_value="hashed_secret",
    ):
        user = await crud.create_user(mock_db, email, password, full_name)

        assert user.email == email
        assert user.hashed_password == "hashed_secret"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_update_portfolio_cash(mock_db):
    pid = uuid4()
    mock_db.execute.return_value = mock_db.mock_result_cls()

    success = await crud.update_portfolio_cash(mock_db, pid, Decimal("100.00"), "add")
    assert success is True

    success_sub = await crud.update_portfolio_cash(
        mock_db, pid, Decimal("50.00"), "subtract"
    )
    assert success_sub is True


@pytest.mark.asyncio
async def test_bulk_create_positions_fallback(mock_db):
    # Test fallback path (no copy_records_to_table)
    mock_conn = AsyncMock()
    mock_conn.get_raw_connection.return_value.driver_connection = MagicMock(
        spec=[]
    )  # No copy method
    mock_db.connection.return_value = mock_conn

    positions_data = [{"symbol": "AAPL", "quantity": 10}]

    count = await crud.bulk_create_positions(mock_db, positions_data)
    assert count == 1
    # Verify execute was called with insert statement
    assert mock_db.execute.call_count == 1


@pytest.mark.asyncio
async def test_bulk_create_positions_fast_path(mock_db):
    # Test fast path (asyncpg copy)
    mock_conn = AsyncMock()
    driver_conn = MagicMock()
    driver_conn.copy_records_to_table = AsyncMock()
    mock_conn.get_raw_connection.return_value.driver_connection = driver_conn
    mock_db.connection.return_value = mock_conn

    positions_data = [{"symbol": "AAPL", "quantity": 10}]

    count = await crud.bulk_create_positions(mock_db, positions_data)
    assert count == 1
    driver_conn.copy_records_to_table.assert_called_once()


@pytest.mark.asyncio
async def test_get_portfolio_summary_mv(mock_db):
    # Test materialized view query
    mock_row = MagicMock()
    mock_row._mapping = {"total_value": 1000}
    mock_db.execute.return_value = [mock_row]

    summary = await crud.get_portfolio_summary(mock_db, uuid4())
    assert len(summary) == 1
    assert summary[0]["total_value"] == 1000


@pytest.mark.asyncio
async def test_get_user_trading_stats(mock_db):
    mock_row = MagicMock()
    mock_row._mapping = {
        "total_orders": 10,
        "filled_orders": 5,
        "cancelled_orders": 2,
        "avg_fill_price": Decimal("100.50"),
    }

    # Mock result.first()
    mock_res = MagicMock()
    mock_res.first.return_value = mock_row
    mock_db.execute.return_value = mock_res

    stats = await crud.get_user_trading_stats(mock_db, uuid4())
    assert stats["total_orders"] == 10
    assert stats["fill_rate"] == 50.0
    assert stats["avg_fill_price"] == 100.5


@pytest.mark.asyncio
async def test_get_iv_surface(mock_db):
    mock_row = MagicMock()
    mock_row._mapping = {"time": datetime.now(), "avg_iv": 0.2}
    mock_db.execute.return_value = [mock_row]

    surface = await crud.get_iv_surface(mock_db, "AAPL")
    assert len(surface) == 1
    assert surface[0]["avg_iv"] == 0.2


@pytest.mark.asyncio
async def test_user_updates(mock_db):
    uid = uuid4()
    mock_db.execute.return_value = mock_db.mock_result_cls()

    await crud.update_user_last_login(mock_db, uid)
    assert mock_db.execute.call_count == 1

    await crud.update_user_tier(mock_db, uid, "pro")
    assert mock_db.execute.call_count == 2  # +1


@pytest.mark.asyncio
async def test_get_positions(mock_db):
    # Mock for get_open_positions_by_portfolio
    mock_pos = Position(symbol="AAPL", quantity=10)
    mock_db.execute.return_value = mock_db.mock_result_cls(items=[mock_pos])

    positions = await crud.get_open_positions_by_portfolio(mock_db, uuid4())
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"

    # Mock for get_expiring_positions
    expiring = await crud.get_expiring_positions(mock_db)
    assert len(expiring) == 1


@pytest.mark.asyncio
async def test_create_order(mock_db):
    mock_db.execute.return_value = mock_db.mock_result_cls()

    order = await crud.create_order(
        mock_db, uuid4(), uuid4(), "AAPL", "buy", 10, "market"
    )
    assert order.symbol == "AAPL"
    assert order.status == "pending"
    mock_db.add.assert_called()


@pytest.mark.asyncio
async def test_ml_model_ops(mock_db):
    # Mock get_latest_model_version returning None
    mock_db.execute.return_value = mock_db.mock_result_cls(scalar=None)

    model = await crud.create_model(mock_db, "price_predictor", "xgboost")
    assert int(model.version) == 1

    # Test set_production_model
    # First query finds model
    mock_db.execute.side_effect = [
        mock_db.mock_result_cls(scalar=model),
        mock_db.mock_result_cls(),
    ]
    success = await crud.set_production_model(mock_db, model.id)
    assert success is True
    assert model.is_production is True
