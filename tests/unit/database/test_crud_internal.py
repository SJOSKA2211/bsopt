import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock
from src.database.models import Base, User, Portfolio, Position, Order, MLModel, OptionPrice
from src.database import crud
from uuid import uuid4
from decimal import Decimal
from datetime import datetime, date, timezone, timedelta

# Use SQLite in-memory for unit testing CRUD logic
@pytest.fixture(name="db_session")
def fixture_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_user_crud(db_session):
    # Create
    user = crud.create_user(db_session, "test@example.com", "password123", "Test User")
    assert user.email == "test@example.com"
    
    # Get by ID
    fetched = crud.get_user_by_id(db_session, user.id)
    assert fetched.id == user.id
    
    # Get by email
    fetched_email = crud.get_user_by_email(db_session, "test@example.com")
    assert fetched_email.id == user.id
    
    # Get with portfolios
    fetched_wp = crud.get_user_with_portfolios(db_session, user.id)
    assert fetched_wp.id == user.id
    
    # Update last login
    crud.update_user_last_login(db_session, user.id)
    assert fetched.last_login is not None
    
    # Update tier
    crud.update_user_tier(db_session, user.id, "pro")
    assert fetched.tier == "pro"
    
    # Get active by tier
    active_users = crud.get_active_users_by_tier(db_session, "pro")
    assert len(active_users) == 1

def test_portfolio_crud(db_session):
    user = crud.create_user(db_session, "test@example.com", "pass", "User")
    
    # Create
    p = crud.create_portfolio(db_session, user.id, "Main", Decimal("1000.00"))
    assert p.name == "Main"
    
    # Get by ID
    fetched = crud.get_portfolio_by_id(db_session, p.id)
    assert fetched.id == p.id
    
    # Get user portfolios
    ps = crud.get_user_portfolios(db_session, user.id)
    assert len(ps) == 1
    
    # Get with open positions
    p_open = crud.get_portfolio_with_open_positions(db_session, p.id)
    assert p_open.id == p.id
    
    # Update cash - add
    success = crud.update_portfolio_cash(db_session, p.id, Decimal("500.00"), "add")
    assert success is True
    assert fetched.cash_balance == Decimal("1500.00")
    
    # Update cash - subtract success
    success = crud.update_portfolio_cash(db_session, p.id, Decimal("200.00"), "subtract")
    assert success is True
    assert fetched.cash_balance == Decimal("1300.00")
    
    # Update cash - subtract failure (insufficient funds)
    success = crud.update_portfolio_cash(db_session, p.id, Decimal("2000.00"), "subtract")
    assert success is False
    assert fetched.cash_balance == Decimal("1300.00")

def test_position_crud(db_session):
    user = crud.create_user(db_session, "u@ex.com", "p", "U")
    p = crud.create_portfolio(db_session, user.id, "P")
    
    # Create
    pos = crud.create_position(db_session, p.id, "AAPL", 10, Decimal("150.00"), option_type="call")
    assert pos.symbol == "AAPL"
    
    # Get by ID
    fetched = crud.get_position_by_id(db_session, pos.id)
    assert fetched.id == pos.id
    
    # Get open by portfolio
    opens = crud.get_open_positions_by_portfolio(db_session, p.id)
    assert len(opens) == 1
    
    # Get by symbol
    syms = crud.get_positions_by_symbol(db_session, "AAPL", status="open")
    assert len(syms) == 1
    
    # Close
    closed = crud.close_position(db_session, pos.id, Decimal("160.00"))
    assert closed.status == "closed"
    assert closed.realized_pnl == Decimal("100.00")
    
    # Close non-existent
    assert crud.close_position(db_session, uuid4(), Decimal("100.00")) is None
    
    # Bulk create
    count = crud.bulk_create_positions(db_session, [
        {"portfolio_id": p.id, "symbol": "MSFT", "quantity": 5, "entry_price": Decimal("300.00"), "status": "open"}
    ])
    assert count == 1
    assert crud.bulk_create_positions(db_session, []) == 0

def test_expiring_positions(db_session):
    user = crud.create_user(db_session, "u2@ex.com", "p", "U")
    p = crud.create_portfolio(db_session, user.id, "P")
    
    today = date.today()
    crud.create_position(db_session, p.id, "AAPL", 10, Decimal("100.00"), expiry=today + timedelta(days=2))
    crud.create_position(db_session, p.id, "MSFT", 10, Decimal("100.00"), expiry=today + timedelta(days=10))
    
    expiring = crud.get_expiring_positions(db_session, days_until_expiry=5)
    assert len(expiring) == 1
    assert expiring[0].symbol == "AAPL"

def test_order_crud(db_session):
    user = crud.create_user(db_session, "u3@ex.com", "p", "U")
    p = crud.create_portfolio(db_session, user.id, "P")
    
    # Create
    o = crud.create_order(db_session, user.id, p.id, "AAPL", "buy", 100, "limit", limit_price=Decimal("150.00"))
    assert o.status == "pending"
    
    # Get by ID
    fetched = crud.get_order_by_id(db_session, o.id)
    assert fetched.id == o.id
    
    # Get user orders
    os_list = crud.get_user_orders(db_session, user.id, status="pending")
    assert len(os_list) == 1
    
    # Get pending orders
    pendings = crud.get_pending_orders(db_session)
    assert len(pendings) >= 1
    
    # Get by broker
    crud.update_order_status(db_session, o.id, "filled", filled_quantity=100, filled_price=Decimal("149.50"))
    # Set broker info manually since create_order doesn't take it
    db_session.execute(update(Order).where(Order.id == o.id).values(broker="IBKR", broker_order_id="BROK123"))
    db_session.commit()
    
    br_os = crud.get_orders_by_broker(db_session, "IBKR", "BROK123")
    assert len(br_os) == 1
    
    # Update status failure
    assert crud.update_order_status(db_session, uuid4(), "cancelled") is False

def test_ml_model_crud(db_session):
    # Create
    m1 = crud.create_model(db_session, "Pricer", "xgboost")
    assert m1.version == 1
    
    m2 = crud.create_model(db_session, "Pricer", "xgboost")
    assert m2.version == 2
    
    # Set production
    crud.set_production_model(db_session, m2.id)
    assert m2.is_production is True
    
    # Get production
    prod = crud.get_production_model(db_session, "Pricer")
    assert prod.id == m2.id
    
    # Set production non-existent
    assert crud.set_production_model(db_session, uuid4()) is False

def test_option_price_queries(db_session):
    expiry = date.today() + timedelta(days=30)
    now = datetime.now(timezone.utc)
    
    op = OptionPrice(
        time=now, symbol="AAPL", strike=Decimal("150.00"), 
        expiry=expiry, option_type="call", bid=Decimal("10.00"), ask=Decimal("11.00")
    )
    db_session.add(op)
    db_session.commit()
    
    # Get latest
    latest = crud.get_latest_option_price(db_session, "AAPL", Decimal("150.00"), expiry, "call")
    assert latest is not None
    assert latest.symbol == "AAPL"
    
    # Get chain with option_type
    chain = crud.get_option_chain(db_session, "AAPL", expiry, option_type="call")
    assert len(chain) == 1

def test_bulk_insert_option_prices_mock():
    # Since it uses postgresql_insert, we mock the session.execute
    mock_session = MagicMock()
    prices_data = [{"symbol": "AAPL"}]
    
    count = crud.bulk_insert_option_prices(mock_session, prices_data)
    assert count == 1
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()
    
    # Empty data
    assert crud.bulk_insert_option_prices(mock_session, []) == 0

def test_get_portfolio_summary_non_existent(db_session):
    summary = crud.get_portfolio_summary(db_session, uuid4())
    assert summary["name"] is None
    assert summary["total_value"] == 0.0

def test_get_user_trading_stats_empty(db_session):
    # Mocking first() to return None to cover the unreachable branch
    with patch.object(db_session, 'execute') as mock_execute:
        mock_execute.return_value.first.return_value = None
        stats = crud.get_user_trading_stats(db_session, uuid4())
        assert stats["total_orders"] == 0

def test_summaries(db_session):
    user = crud.create_user(db_session, "summary@ex.com", "p", "U")
    p = crud.create_portfolio(db_session, user.id, "P", Decimal("1000.00"))
    
    crud.create_position(db_session, p.id, "AAPL", 10, Decimal("100.00")) # status defaults to open
    pos2 = crud.create_position(db_session, p.id, "MSFT", 5, Decimal("200.00"))
    crud.close_position(db_session, pos2.id, Decimal("210.00"))
    
    summary = crud.get_portfolio_summary(db_session, p.id)
    assert summary["open_positions"] == 1
    assert summary["total_realized_pnl"] == 50.0
    
    # Trading stats
    crud.create_order(db_session, user.id, p.id, "AAPL", "buy", 100, "market")
    stats = crud.get_user_trading_stats(db_session, user.id)
    assert stats["total_orders"] == 1