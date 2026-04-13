from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.database import crud
from src.database.models import Base, User


# Setup in-memory sqlite for fast CRUD testing
@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_user_crud(db_session):
    email = "test@example.com"
    with patch("src.database.crud.password_service.hash_password", return_value="hashed"):
        user = await crud.create_user(db_session, email, "pass", "Full Name")
        assert user.email == email

    retrieved = await crud.get_user_by_email(db_session, email)
    assert retrieved.id == user.id

    await crud.update_user_tier(db_session, user.id, "enterprise")
    updated = await crud.get_user_by_id(db_session, user.id)
    assert updated.tier == "enterprise"


@pytest.mark.asyncio
async def test_portfolio_crud(db_session):
    user = User(email="p@ex.com", hashed_password="h", full_name="N")
    db_session.add(user)
    await db_session.commit()

    port = await crud.create_portfolio(db_session, user.id, "Growth", Decimal("1000.00"))
    assert port.name == "Growth"
    assert port.cash_balance == 1000.0

    success = await crud.update_portfolio_cash(db_session, port.id, Decimal("500.00"), "add")
    assert success is True
    await db_session.refresh(port)
    assert port.cash_balance == 1500.0


@pytest.mark.asyncio
async def test_position_crud(db_session):
    user = User(email="pos@ex.com", hashed_password="h", full_name="N")
    db_session.add(user)
    await db_session.commit()
    port = await crud.create_portfolio(db_session, user.id, "P1")

    pos = await crud.create_position(db_session, port.id, "AAPL", 10, Decimal("150.00"))
    assert pos.symbol == "AAPL"

    # Close position
    closed = await crud.close_position(db_session, pos.id, Decimal("160.00"))
    assert closed.status == "closed"
    assert closed.realized_pnl == 100.0  # (160-150)*10


@pytest.mark.asyncio
async def test_order_crud(db_session):
    user = User(email="ord@ex.com", hashed_password="h", full_name="N")
    db_session.add(user)
    await db_session.commit()
    port = await crud.create_portfolio(db_session, user.id, "P1")

    order = await crud.create_order(
        db_session, user.id, port.id, "AAPL", "buy", 5, "limit", Decimal("145.00")
    )
    assert order.status == "pending"

    await crud.update_order_status(
        db_session,
        order.id,
        "filled",
        filled_quantity=5,
        filled_price=Decimal("145.00"),
    )
    await db_session.refresh(order)
    assert order.status == "filled"


@pytest.mark.asyncio
async def test_ml_model_crud(db_session):
    model = await crud.create_model(db_session, "AlphaV1", "xgboost")
    assert str(model.version) == "1"

    model2 = await crud.create_model(db_session, "AlphaV1", "xgboost")
    assert str(model2.version) == "2"

    await crud.set_production_model(db_session, model2.id)
    await db_session.refresh(model2)
    assert model2.is_production is True