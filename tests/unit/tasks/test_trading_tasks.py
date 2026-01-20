import pytest
from unittest.mock import MagicMock, patch
from src.tasks.trading_tasks import execute_trade_task, backtest_strategy_task, check_risk_limits

def test_check_risk_limits():
    assert check_risk_limits({"quantity": 10, "limit_price": 100}) is True
    assert check_risk_limits({"quantity": 1000, "limit_price": 200}) is False

@patch("src.tasks.trading_tasks.celery_app.Task.request")
def test_execute_trade_task_success(mock_request):
    order = {"symbol": "AAPL", "quantity": 10, "limit_price": 150.0, "side": "buy"}
    mock_request.id = "test-id-12345678"
    
    with patch("time.sleep"), patch("random.uniform", return_value=1.0):
        result = execute_trade_task.apply(args=[order], task_id=mock_request.id).get()
        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"

@patch("src.tasks.trading_tasks.celery_app.Task.request")
def test_execute_trade_task_risk_reject(mock_request):
    order = {"symbol": "AAPL", "quantity": 1000, "limit_price": 200.0}
    mock_request.id = "test-id"
    
    result = execute_trade_task.apply(args=[order], task_id=mock_request.id).get()
    assert result["status"] == "rejected"
    assert result["reason"] == "risk_limit_exceeded"

@patch("src.tasks.trading_tasks.celery_app.Task.request")
def test_execute_trade_task_invalid(mock_request):
    order = {} # Missing params
    mock_request.id = "test-id"
    
    result = execute_trade_task.apply(args=[order], task_id=mock_request.id).get()
    assert result["status"] == "failed"
    assert "Invalid order parameters" in result["error"]

@patch("src.tasks.trading_tasks.celery_app.Task.request")
def test_backtest_strategy_task_success(mock_request):
    strategy = "Mean Reversion"
    mock_request.id = "backtest-id"
    
    with patch("time.sleep"), patch("random.uniform", return_value=1.0), patch("random.randint", return_value=100):
        result = backtest_strategy_task.apply(args=[strategy, "2023-01-01", "2023-12-31"], task_id=mock_request.id).get()
        assert result["status"] == "completed"
        assert result["strategy"] == strategy

@patch("src.tasks.trading_tasks.celery_app.Task.request")
def test_backtest_strategy_task_error(mock_request):
    mock_request.id = "backtest-id"
    with patch("time.sleep", side_effect=Exception("Simulated error")):
        result = backtest_strategy_task.apply(args=["strategy", "2023-01-01", "2023-12-31"], task_id=mock_request.id).get()
        assert result["status"] == "failed"
        assert result["error"] == "Simulated error"