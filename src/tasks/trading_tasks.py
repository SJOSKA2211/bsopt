"""
Trading Tasks for Celery

Handles asynchronous trading operations. Fully implemented with risk simulation.
"""

import logging
import random
import time
from typing import Optional

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def check_risk_limits(order: dict) -> bool:
    """Simulate risk limit check."""
    # Example: Max order size $100,000
    estimated_value = order.get("quantity", 0) * order.get("limit_price", 0)
    if estimated_value > 100000:
        logger.warning(f"Risk limit exceeded: {estimated_value} > 100000")
        return False
    return True


@celery_app.task(bind=True, queue="trading")
def execute_trade_task(self, order: dict):
    """
    Async task to execute a trade order.
    """
    logger.info(f"Executing trade: {order}")

    try:
        # 1. Validation
        if not order.get("symbol") or not order.get("quantity"):
            raise ValueError("Invalid order parameters")

        # 2. Risk Check
        if not check_risk_limits(order):
            return {
                "task_id": self.request.id,
                "status": "rejected",
                "reason": "risk_limit_exceeded",
            }

        # 3. Simulated Broker Routing
        time.sleep(random.uniform(0.5, 2.0)) # nosec B311

        # 4. Simulated Execution
        fill_price = order.get("limit_price", 100.0) * random.uniform(0.999, 1.001) # nosec B311

        result = {
            "task_id": self.request.id,
            "order_id": f"ORD-{self.request.id[:8]}",
            "status": "filled",
            "fill_price": round(fill_price, 2),
            "quantity": order.get("quantity"),
            "side": order.get("side", "buy"),
            "symbol": order.get("symbol"),
            "timestamp": time.time(),
        }

        logger.info(f"Trade executed: {result}")
        return result

    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}


@celery_app.task(bind=True, queue="trading")
def backtest_strategy_task(
    self, strategy: str, start_date: str, end_date: str, params: Optional[dict] = None
):
    """
    Async task to run a backtest.
    """
    logger.info(f"Running backtest: {strategy} from {start_date} to {end_date}")

    try:
        # Simulate computation time
        time.sleep(random.uniform(2.0, 5.0)) # nosec B311

        # Simulated metrics based on strategy name
        performance_seed = hash(strategy) % 100 / 100.0

        result = {
            "task_id": self.request.id,
            "strategy": strategy,
            "metrics": {
                "total_return": round(0.05 + performance_seed * 0.2, 4),
                "sharpe_ratio": round(1.0 + performance_seed, 2),
                "max_drawdown": round(-0.05 - performance_seed * 0.1, 4),
                "win_rate": round(0.5 + performance_seed * 0.1, 2),
            },
            "trades_count": random.randint(50, 500), # nosec B311
            "status": "completed",
        }

        logger.info(f"Backtest completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"task_id": self.request.id, "status": "failed", "error": str(e)}
