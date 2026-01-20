import pytest
from datetime import datetime, date, timezone
from decimal import Decimal
from uuid import uuid4
from src.database.models import (
    User, APIKey, OptionPrice, MarketTick, Portfolio, Position,
    Order, MLModel, ModelPrediction, RateLimit, AuditLog,
    RequestLog, SecurityIncident, GDPRRequest, CalibrationResult
)

def test_user_repr():
    u = User(id=uuid4(), email="test@example.com", tier="pro")
    assert "test@example.com" in repr(u)
    assert "pro" in repr(u)

def test_apikey_repr():
    ak = APIKey(name="TestKey", prefix="bs_123")
    assert "TestKey" in repr(ak)
    assert "bs_123" in repr(ak)

def test_option_price_logic():
    op = OptionPrice(
        symbol="AAPL", strike=Decimal("150.00"), option_type="call",
        time=datetime.now(timezone.utc), bid=Decimal("10.00"), ask=Decimal("11.00")
    )
    assert op.mid_price == Decimal("10.50")
    assert "AAPL" in repr(op)
    
    op_no_bid = OptionPrice(bid=None, ask=Decimal("11.00"))
    assert op_no_bid.mid_price is None

def test_market_tick_repr():
    mt = MarketTick(symbol="AAPL", time=datetime.now(timezone.utc), price=Decimal("150.00"))
    assert "AAPL" in repr(mt)
    assert "150.00" in repr(mt)

def test_portfolio_logic():
    p = Portfolio(id=uuid4(), name="MyPortfolio", cash_balance=Decimal("1000.00"))
    pos1 = Position(status="open", quantity=10, entry_price=Decimal("10.00"))
    pos2 = Position(status="closed", quantity=5, entry_price=Decimal("20.00"))
    p.positions = [pos1, pos2]
    
    assert len(p.open_positions) == 1
    assert p.total_value == Decimal("1100.00")
    assert "MyPortfolio" in repr(p)

def test_position_logic():
    pos = Position(symbol="AAPL", quantity=10, entry_price=Decimal("100.00"), status="open", option_type="call")
    assert pos.is_option is True
    assert pos.unrealized_pnl is None
    assert "AAPL" in repr(pos)
    
    pos.close(exit_price=Decimal("110.00"))
    assert pos.status == "closed"
    assert pos.realized_pnl == Decimal("100.00")
    assert pos.exit_date is not None
    
    pos_no_opt = Position(option_type=None)
    assert pos_no_opt.is_option is False

def test_order_repr():
    o = Order(side="buy", quantity=100, symbol="AAPL", order_type="market")
    assert "buy" in repr(o)
    assert "AAPL" in repr(o)

def test_ml_model_repr():
    m = MLModel(name="PriceModel", version=1, algorithm="xgboost")
    assert "PriceModel" in repr(m)
    assert "v1" in repr(m)

def test_model_prediction_logic():
    mp = ModelPrediction(predicted_price=Decimal("10.50"), actual_price=Decimal("10.00"))
    assert mp.calculate_error() == Decimal("0.50")
    assert "10.50" in repr(mp)
    
    mp_no_actual = ModelPrediction(actual_price=None)
    assert mp_no_actual.calculate_error() is None

def test_rate_limit_repr():
    rl = RateLimit(user_id=uuid4(), endpoint="/test", request_count=5)
    assert "/test" in repr(rl)
    assert "5" in repr(rl)

def test_audit_log_repr():
    al = AuditLog(event_type="LOGIN", user_id=uuid4(), created_at=datetime.now(timezone.utc))
    assert "LOGIN" in repr(al)

def test_request_log_repr():
    rl = RequestLog(method="GET", path="/api", status_code=200, response_time_ms=Decimal("50.5"))
    assert "GET" in repr(rl)
    assert "200" in repr(rl)
    assert "50.5" in repr(rl)

def test_security_incident_repr():
    si = SecurityIncident(incident_type="BRUTE_FORCE", detected_at=datetime.now(timezone.utc))
    assert "BRUTE_FORCE" in repr(si)

def test_gdpr_request_repr():
    gr = GDPRRequest(user_id=uuid4(), request_type="DELETE", status="pending")
    assert "DELETE" in repr(gr)
    assert "pending" in repr(gr)

def test_calibration_result_repr():
    cr = CalibrationResult(symbol="AAPL", time=datetime.now(timezone.utc), rmse=0.001)
    assert "AAPL" in repr(cr)
    assert "0.001" in repr(cr)
