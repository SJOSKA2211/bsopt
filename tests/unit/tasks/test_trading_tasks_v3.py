from unittest.mock import MagicMock, patch

from src.workers.tasks.trading_tasks import (
    backtest_strategy_task,
    check_risk_limits,
    execute_trade_task,
)


def test_check_risk_limits():
    assert check_risk_limits({"quantity": 10, "limit_price": 100}) is True
    assert not check_risk_limits({"quantity": 1000, "limit_price": 200})


def test_execute_trade_task_success():
    # Dig deep to find the naked function
    func = execute_trade_task
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    mock_self = MagicMock()
    mock_self.request.id = "test-id"
    order = {"symbol": "AAPL", "quantity": 10, "limit_price": 150.0, "side": "buy"}

    with patch("time.sleep"):
        res = func(mock_self, order)
        assert res["status"] == "filled"


def test_execute_trade_task_invalid():
    func = execute_trade_task
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    mock_self = MagicMock()
    mock_self.request.id = "test-id"
    res = func(mock_self, {})
    assert res["status"] == "failed"


def test_backtest_strategy_task_success():
    func = backtest_strategy_task
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    mock_self = MagicMock()
    mock_self.request.id = "test-id"

    with patch("src.workers.tasks.trading_tasks.BacktestEngine") as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.run_vectorized.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "duration_seconds": 1.5,
            "trades_count": 42,
        }

        res = func(
            mock_self,
            strategy="momentum",
            start_date="2025-01-01",
            end_date="2025-01-10",
        )
        assert res["status"] == "completed"
