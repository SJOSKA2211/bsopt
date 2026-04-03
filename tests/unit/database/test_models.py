from datetime import datetime

from src.database.models import MarketTick, OptionPrice, User


def test_models_creation():
    # Just test instantiation and basic attributes
    user = User(email="test@example.com", full_name="Test User")
    assert user.email == "test@example.com"

    trade = OptionPrice(symbol="AAPL", last=150.0, time=datetime.now())
    assert trade.symbol == "AAPL"
    assert trade.last == 150.0

    md = MarketTick(symbol="AAPL", price=150.0, time=datetime.now())
    assert md.symbol == "AAPL"


def test_user_instantiation():
    user = User(email="user@test.com", full_name="User Test", is_active=True)
    assert user.email == "user@test.com"
    assert user.is_active is True
