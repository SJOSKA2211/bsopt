import pytest
from unittest.mock import MagicMock, patch
from src.tasks.trading_tasks import execute_trade_task, backtest_strategy_task

@pytest.fixture
def mock_celery():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    return celery_app

@patch("src.tasks.trading_tasks.time.sleep")
def test_execute_trade_success(mock_sleep, mock_celery):
    order = {
        "symbol": "AAPL",
        "quantity": 100,
        "limit_price": 150.0,
        "side": "buy"
    }
    res = execute_trade_task.apply(args=[order])
    result = res.get()
    
    assert result["status"] == "filled"
    assert result["symbol"] == "AAPL"
    assert result["quantity"] == 100
    mock_sleep.assert_called()

@patch("src.tasks.trading_tasks.time.sleep")
def test_execute_trade_risk_fail(mock_sleep, mock_celery):
    # Risk limit is 100,000
    order = {
        "symbol": "BRk-A",
        "quantity": 1000,
        "limit_price": 500000.0, # 500M > 100k
        "side": "buy"
    }
    res = execute_trade_task.apply(args=[order])
    result = res.get()
    
    assert result["status"] == "rejected"
    assert result["reason"] == "risk_limit_exceeded"

@patch("src.tasks.trading_tasks.time.sleep")
def test_execute_trade_invalid_input(mock_sleep, mock_celery):
    order = {
        "quantity": 100
        # Missing symbol
    }
    res = execute_trade_task.apply(args=[order])
    result = res.get()
    
    assert result["status"] == "failed"
    assert "Invalid order parameters" in result["error"]

@patch("src.tasks.trading_tasks.time.sleep")
def test_backtest_strategy_task(mock_sleep, mock_celery):
    res = backtest_strategy_task.apply(args=["momentum", "2023-01-01", "2023-12-31"])
    result = res.get()
    
    assert result["status"] == "completed"
    assert result["strategy"] == "momentum"
    assert "metrics" in result
    assert result["metrics"]["sharpe_ratio"] > 0

@patch("src.tasks.trading_tasks.time.sleep")
def test_backtest_error(mock_sleep, mock_celery):
    # Pass invalid inputs to trigger exception or mock it
    # Just force exception by patching internal logic or something?
    # Or maybe mock time.sleep to raise?
    
    mock_sleep.side_effect = Exception("Backtest crash")
    res = backtest_strategy_task.apply(args=["momentum", "2023-01-01", "2023-12-31"])
    result = res.get()
    
    assert result["status"] == "failed"
    assert "Backtest crash" in result["error"]
