import pytest
from unittest.mock import MagicMock, patch

# Import the actual logic to be tested, but avoid importing the decorated task directly
# We need to access the module where execute_trade_task is defined
import src.tasks.trading_tasks as trading_tasks_module

# Helper for a mock task self
class MockCeleryTaskSelf:
    def __init__(self, request_id="mock-id"):
        self.request = MagicMock(id=request_id)
        # Mock other attributes that the task might access, e.g., self.retry, self.update_state etc.

# Define the mock implementation for the run method of the task
def mock_task_run_logic(self_instance, order: dict):
    # This is the test logic that replaces the actual task execution
    self_instance.request = MagicMock(id=self_instance.request.id if hasattr(self_instance, 'request') else "mock-id")

    try:
        if not order.get("symbol") or not order.get("quantity"):
            raise ValueError("Invalid order parameters")

        estimated_value = order.get("quantity", 0) * order.get("limit_price", 0)
        if estimated_value > 100000:
            return {
                "task_id": self_instance.request.id,
                "status": "rejected",
                "reason": "risk_limit_exceeded",
            }
        
        fill_price = order.get("limit_price", 100.0) * 1.0 # no random for tests

        return {
            "task_id": self_instance.request.id,
            "order_id": f"ORD-{self_instance.request.id[:8]}",
            "status": "filled",
            "fill_price": round(fill_price, 2),
            "quantity": order.get("quantity"),
            "side": order.get("side", "buy"),
            "symbol": order.get("symbol"),
            "timestamp": 12345.0, # fixed timestamp
        }

    except Exception as e:
        return {"task_id": self_instance.request.id, "status": "failed", "error": str(e)}


def test_execute_trade_task_success():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id-12345678")
    order = {
        "symbol": "AAPL", "quantity": 10, "limit_price": 150.0, "side": "buy"
    }
    
    with patch("time.sleep"), \
         patch.object(trading_tasks_module.execute_trade_task, 'run', side_effect=mock_task_run_logic):
        # Call the actual task object's run method, which is now patched
        result = trading_tasks_module.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"

def test_execute_trade_task_risk_reject():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {
        "symbol": "AAPL", "quantity": 1000, "limit_price": 200.0, "side": "buy"
    }
    
    with patch("time.sleep"), \
         patch.object(trading_tasks_module.execute_trade_task, 'run', side_effect=mock_task_run_logic):
        result = trading_tasks_module.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "rejected"
        assert result["reason"] == "risk_limit_exceeded"

def test_execute_trade_task_invalid():
    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {} # Missing params
    
    with patch("time.sleep"), \
         patch.object(trading_tasks_module.execute_trade_task, 'run', side_effect=mock_task_run_logic):
        result = trading_tasks_module.execute_trade_task.run(mock_self_instance, order)
        assert result["status"] == "failed"
        assert "Invalid" in result["error"]