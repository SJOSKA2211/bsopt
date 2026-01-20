import pytest
from datetime import datetime, timedelta
from src.pricing.graphql.schema import Option, schema
import strawberry

def test_option_fields(mocker):
    expiry = datetime.now() + timedelta(days=365)
    opt = Option(id=strawberry.ID("1"), strike=150.0, underlying_symbol="AAPL", expiry=expiry, option_type="call")
    
    # Mock HybridQuantumClassicalPricer
    mock_hybrid = mocker.patch("src.pricing.graphql.schema.HybridQuantumClassicalPricer")
    mock_instance = mock_hybrid.return_value
    mock_instance.price_option_adaptive.return_value = (10.0, 0.1)
    
    assert opt.price() == 10.0
    assert opt.delta() > 0
    assert opt.gamma() > 0

def test_option_fields_expired(mocker):
    # Case where T <= 0
    expiry = datetime.now() - timedelta(days=1)
    opt = Option(id=strawberry.ID("1"), strike=150.0, underlying_symbol="AAPL", expiry=expiry, option_type="call")
    
    mock_hybrid = mocker.patch("src.pricing.graphql.schema.HybridQuantumClassicalPricer")
    mock_instance = mock_hybrid.return_value
    mock_instance.price_option_adaptive.return_value = (5.0, 0.1)
    
    assert opt.price() == 5.0
    assert opt.delta() is not None
    assert opt.gamma() is not None

def test_resolve_reference():
    expiry_str = "2026-01-17T12:00:00"
    opt = Option.resolve_reference(
        id=strawberry.ID("1"),
        strike=150.0,
        underlyingSymbol="AAPL",
        expiry=expiry_str,
        optionType="call"
    )
    assert opt.id == "1"
    assert opt.strike == 150.0
    assert opt.underlying_symbol == "AAPL"
    assert isinstance(opt.expiry, datetime)
    
    # Already datetime
    expiry_dt = datetime.now()
    opt2 = Option.resolve_reference("2", 100.0, "MSFT", expiry_dt, "put")
    assert opt2.expiry == expiry_dt

def test_query_dummy():
    query = "{ dummy }"
    result = schema.execute_sync(query)
    assert result.data["dummy"] == "pricing"
