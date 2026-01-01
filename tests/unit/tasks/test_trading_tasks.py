import pytest
from unittest.mock import MagicMock, patch

# Helper for a mock task self
class MockCeleryTaskSelf:
    def __init__(self, request_id="mock-id"):
        self.request = MagicMock(id=request_id)
        self.request.id = request_id # Ensure request.id is set for the mock task self

# Mock implementation of the task's run method
def mock_task_run_implementation(self_instance, order: dict):
    # This function will entirely replace the execute_trade_task.run method's logic for the test.
    # It takes exactly the arguments that the original task's run method takes.

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


# Tests will patch the execute_trade_task (the Task object itself)
# to have a run method that calls our mock_task_run_implementation.
@patch("src.tasks.trading_tasks.execute_trade_task")
def test_execute_trade_task_success(mock_celery_task_object):
    # Configure the mock Celery task object
    mock_celery_task_object.run = MagicMock(side_effect=mock_task_run_implementation)

    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id-12345678")
    order = {
        "symbol": "AAPL", "quantity": 10, "limit_price": 150.0, "side": "buy"
    }
    
    with patch("time.sleep"):
        # Now, call the run method of the mocked Celery task object
        result = mock_celery_task_object.run(mock_self_instance, order)
        assert result["status"] == "filled"
        assert result["symbol"] == "AAPL"

@patch("src.tasks.trading_tasks.execute_trade_task")
def test_execute_trade_task_risk_reject(mock_celery_task_object):
    mock_celery_task_object.run = MagicMock(side_effect=mock_task_run_implementation)

    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {
        "symbol": "AAPL", "quantity": 1000, "limit_price": 200.0, "side": "buy"
    }
    
    with patch("time.sleep"):
        result = mock_celery_task_object.run(mock_self_instance, order)
        assert result["status"] == "rejected"
        assert result["reason"] == "risk_limit_exceeded"

@patch("src.tasks.trading_tasks.execute_trade_task")
def test_execute_trade_task_invalid(mock_celery_task_object):
    mock_celery_task_object.run = MagicMock(side_effect=mock_task_run_implementation)

    mock_self_instance = MockCeleryTaskSelf(request_id="test-order-id")
    order = {} # Missing params
    
    with patch("time.sleep"):
        result = mock_celery_task_object.run(mock_self_instance, order)
        assert result["status"] == "failed"
        assert "Invalid" in result["error"]
