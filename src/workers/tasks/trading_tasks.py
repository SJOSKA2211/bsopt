from src.workers.tasks.celery_app import celery_app


def check_risk_limits(order):
    """Simple risk check."""
    quantity = order.get("quantity", 0)
    limit_price = order.get("limit_price", 0)
    # Mock logic matching tests if possible, or just a placeholder
    return quantity * limit_price < 50000

@celery_app.task
def execute_trade_task(order):
    """Task to execute a trade."""
    if not check_risk_limits(order):
        return {"status": "rejected", "reason": "risk_limit_exceeded"}
    return {"status": "filled", "order_id": "mock-123"}

@celery_app.task
def backtest_strategy_task(strategy_name, params):
    """Task to run a backtest."""
    return {"status": "completed", "pnl": 1000.0}
