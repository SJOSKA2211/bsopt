import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import crud
from src.database.models import Base, OptionPrice


@pytest.fixture(autouse=True)
def mock_password_service():
    with patch("src.database.crud.password_service") as mock:
        mock.hash_password.return_value = "hashed_password"
        mock.generate_reset_token.return_value = "token"
        yield mock


@pytest.fixture
def db_session():
    # Force SQLite to accept JSONB as JSON
    engine = create_engine("sqlite:///:memory:")

    # Define a custom compilation for JSONB on SQLite
    from sqlalchemy.dialects import postgresql as pg_dialect
    from sqlalchemy.ext.compiler import compiles

    @compiles(pg_dialect.JSONB, "sqlite")
    def compile_jsonb_sqlite(type_, compiler, **kw):
        return "JSON"

    # Handle PostgreSQL UUIDs in SQLite
    @compiles(pg_dialect.UUID, "sqlite")
    def compile_uuid_sqlite(type_, compiler, **kw):
        return "CHAR(32)"

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_user_operations(db_session):
    user = crud.create_user(db_session, "test@ex.com", "pass", "Test User")
    assert user.email == "test@ex.com"

    # get_user_by_email
    fetched_email = crud.get_user_by_email(db_session, "test@ex.com")
    assert fetched_email.id == user.id

    # get_user_by_id
    fetched = crud.get_user_by_id(db_session, user.id)
    assert fetched.id == user.id

    # get_user_with_portfolios
    fetched_wp = crud.get_user_with_portfolios(db_session, user.id)
    assert fetched_wp.id == user.id

    # update_user_last_login
    crud.update_user_last_login(db_session, user.id)
    db_session.refresh(user)
    assert user.last_login is not None

    # update_user_tier
    crud.update_user_tier(db_session, user.id, "pro")
    db_session.refresh(user)
    assert user.tier == "pro"

    # get_active_users_by_tier
    active_users = crud.get_active_users_by_tier(db_session, "pro")
    assert len(active_users) >= 1


def test_portfolio_operations(db_session):
    user = crud.create_user(db_session, "p@ex.com", "pass", "P User")
    portfolio = crud.create_portfolio(db_session, user.id, "Main", Decimal("1000.00"))

    # get_portfolio_by_id
    p = crud.get_portfolio_by_id(db_session, portfolio.id)
    assert p.id == portfolio.id

    # get_user_portfolios including positions
    ups = crud.get_user_portfolios(db_session, user.id, include_positions=True)
    assert len(ups) == 1

    # get_portfolio_with_open_positions
    p_open = crud.get_portfolio_with_open_positions(db_session, portfolio.id)
    assert p_open.id == portfolio.id

    # get_portfolio_with_open_positions (non-existent)
    assert crud.get_portfolio_with_open_positions(db_session, uuid.uuid4()) is None

    # update_portfolio_cash
    success = crud.update_portfolio_cash(
        db_session, portfolio.id, Decimal("500.00"), "add"
    )
    assert success
    db_session.refresh(portfolio)
    assert portfolio.cash_balance == Decimal("1500.00")

    success = crud.update_portfolio_cash(
        db_session, portfolio.id, Decimal("2000.00"), "subtract"
    )
    assert not success  # Insufficient funds

    success = crud.update_portfolio_cash(
        db_session, portfolio.id, Decimal("500.00"), "subtract"
    )
    assert success
    db_session.refresh(portfolio)
    assert portfolio.cash_balance == Decimal("1000.00")


def test_position_operations(db_session):
    user = crud.create_user(db_session, "pos@ex.com", "pass", "Pos User")
    portfolio = crud.create_portfolio(db_session, user.id, "Main")

    pos = crud.create_position(
        db_session,
        portfolio.id,
        "AAPL",
        10,
        Decimal("150.00"),
        strike=Decimal("155.00"),
        expiry=date.today() + timedelta(days=30),
        option_type="call",
    )

    # get_position_by_id
    fetched = crud.get_position_by_id(db_session, pos.id)
    assert fetched.id == pos.id

    # get_open_positions_by_portfolio
    open_pos = crud.get_open_positions_by_portfolio(db_session, portfolio.id)
    assert len(open_pos) == 1

    # get_expiring_positions
    expiring = crud.get_expiring_positions(db_session, days_until_expiry=40)
    assert len(expiring) >= 1

    # get_positions_by_symbol
    sym_pos = crud.get_positions_by_symbol(db_session, "AAPL", "open")
    assert len(sym_pos) == 1

    # close_position
    closed = crud.close_position(db_session, pos.id, Decimal("160.00"))
    assert closed.status == "closed"
    assert closed.realized_pnl == Decimal("100.00")  # (160-150)*10

    # close_position (already closed or non-existent)
    assert crud.close_position(db_session, pos.id, Decimal("170.00")) is None
    assert crud.close_position(db_session, uuid.uuid4(), Decimal("170.00")) is None

    # bulk_create_positions
    data = [
        {
            "portfolio_id": portfolio.id,
            "symbol": "MSFT",
            "quantity": 5,
            "entry_price": Decimal("300.00"),
            "status": "open",
        },
        {
            "portfolio_id": portfolio.id,
            "symbol": "GOOG",
            "quantity": 2,
            "entry_price": Decimal("2500.00"),
            "status": "open",
        },
    ]
    count = crud.bulk_create_positions(db_session, data)
    assert count == 2

    # bulk_create_positions empty
    assert crud.bulk_create_positions(db_session, []) == 0


def test_order_operations(db_session):
    user = crud.create_user(db_session, "ord@ex.com", "pass", "Ord User")
    portfolio = crud.create_portfolio(db_session, user.id, "Main")

    order = crud.create_order(
        db_session,
        user.id,
        portfolio.id,
        "TSLA",
        "buy",
        5,
        "limit",
        limit_price=Decimal("700.00"),
    )

    # get_order_by_id
    o = crud.get_order_by_id(db_session, order.id)
    assert o.id == order.id

    # get_user_orders
    u_orders = crud.get_user_orders(db_session, user.id, status="pending")
    assert len(u_orders) == 1

    # get_pending_orders
    pending = crud.get_pending_orders(db_session)
    assert len(pending) == 1

    # get_orders_by_broker
    order.broker = "interactive_brokers"
    order.broker_order_id = "B123"
    db_session.commit()
    b_orders = crud.get_orders_by_broker(db_session, "interactive_brokers", "B123")
    assert len(b_orders) == 1

    # update_order_status
    success = crud.update_order_status(
        db_session,
        order.id,
        "filled",
        filled_quantity=5,
        filled_price=Decimal("695.00"),
    )
    assert success
    db_session.refresh(order)
    assert order.status == "filled"


def test_ml_model_operations(db_session):
    model = crud.create_model(db_session, "price_pred", "xgboost")
    assert model.version == 1

    # get_latest_model_version
    latest = crud.get_latest_model_version(db_session, "price_pred")
    assert latest.version == 1

    # create another version
    model2 = crud.create_model(db_session, "price_pred", "xgboost")
    assert model2.version == 2

    # set_production_model
    success = crud.set_production_model(db_session, model2.id)
    assert success
    db_session.refresh(model2)
    assert model2.is_production

    # get_production_model
    prod = crud.get_production_model(db_session, "price_pred")
    assert prod.id == model2.id

    # set_production_model non-existent
    assert not crud.set_production_model(db_session, uuid.uuid4())


def test_option_price_operations(db_session):
    expiry = date.today() + timedelta(days=30)
    # Since bulk_insert_option_prices uses PostgreSQL dialect specific code,
    # we'll mock the execution to get coverage without failing on SQLite
    with patch.object(db_session, "execute") as mock_exec:
        crud.bulk_insert_option_prices(db_session, [{"symbol": "AAPL"}])
        mock_exec.assert_called()

    # get_latest_option_price (manual insert first)
    op = OptionPrice(
        time=datetime.now(UTC),
        symbol="AAPL",
        strike=Decimal("150.00"),
        expiry=expiry,
        option_type="call",
        bid=Decimal("10.00"),
        ask=Decimal("11.00"),
        last=Decimal("10.50"),
    )
    db_session.add(op)
    db_session.commit()

    fetched = crud.get_latest_option_price(
        db_session, "AAPL", Decimal("150.00"), expiry, "call"
    )
    assert fetched.symbol == "AAPL"

    # get_option_chain
    chain = crud.get_option_chain(db_session, "AAPL", expiry, option_type="call")
    assert len(chain) == 1


def test_aggregation_queries(db_session):
    user = crud.create_user(db_session, "agg@ex.com", "pass", "Agg User")
    portfolio = crud.create_portfolio(db_session, user.id, "Main", Decimal("1000.00"))

    pos = crud.create_position(db_session, portfolio.id, "AAPL", 10, Decimal("100.00"))
    db_session.commit()

    # get_portfolio_summary
    summary = crud.get_portfolio_summary(db_session, portfolio.id)
    assert summary["portfolio_id"] == portfolio.id
    assert summary["open_positions"] == 1
    assert summary["open_position_value"] == 1000.0
    assert summary["total_value"] == 2000.0

    # get_portfolio_summary (non-existent portfolio)
    summary_empty = crud.get_portfolio_summary(db_session, uuid.uuid4())
    assert summary_empty["name"] is None

    # get_user_trading_stats
    stats = crud.get_user_trading_stats(db_session, user.id)
    assert stats["total_orders"] == 0
