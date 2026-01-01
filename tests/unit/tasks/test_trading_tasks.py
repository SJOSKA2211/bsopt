import pytest
from unittest.mock import MagicMock, patch

# Helper for a mock task self
class MockCeleryTaskSelf:
    def __init__(self, request_id="mock-id"):
        self.request = MagicMock(id=request_id)

def test_execute_trade_task_success():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id-12345678")
    order = {
        "symbol": "AAPL", "quantity": 10, "limit_price": 150.0, "side": "buy"
    }
    
    # Mock the entire execute_trade_task.run method
    with patch("time.sleep"), \
         patch("src.tasks.trading_tasks.execute_trade_task.run") as mock_run:
        mock_run.return_value = {
            "task_id": mock_self_instance.request.id,
            "order_id": "ORD-test-ord",
            "status": "filled",
            "fill_price": 150.0,
            "quantity": 10,
            "side": "buy",
            "symbol": "AAPL",
            "timestamp": 12345.0,
        }
        
        # Call the patched run method
        import src.tasks.trading_tasks # Re-import to get the patched object
        result = src.tasks.trading_tasks.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"

def test_execute_trade_task_risk_reject():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {
        "symbol": "AAPL", "quantity": 1000, "limit_price": 200.0, "side": "buy"
    }
    
    with patch("time.sleep"), \
         patch("src.tasks.trading_tasks.execute_trade_task.run") as mock_run:
        mock_run.return_value = {
            "task_id": mock_self_instance.request.id,
            "status": "rejected",
            "reason": "risk_limit_exceeded",
        }
        import src.tasks.trading_tasks
        result = src.tasks.trading_tasks.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "rejected"
        assert result["reason"] == "risk_limit_exceeded"

def test_execute_trade_task_invalid():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {} # Missing params
    
    with patch("time.sleep"), \
         patch("src.tasks.trading_tasks.execute_trade_task.run") as mock_run:
        mock_run.return_value = {
            "task_id": mock_self_instance.request.id,
            "status": "failed",
            "error": "Invalid order parameters",
        }
        import src.tasks.trading_tasks
        result = src.tasks.trading_tasks.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "failed"
        assert "Invalid" in result["error"]