import logging
import uuid
import asyncio
from decimal import Decimal
from datetime import datetime, timezone

import structlog
import numpy as np
import yfinance as yf
from sqlalchemy import select, update

from src.database import db_manager
from src.database.models import Order, Portfolio
from src.workers.tasks.celery_app import celery_app
from src.shared.trading.broker import get_broker

logger = structlog.get_logger(__name__)


async def check_risk_limits_async(order_id: uuid.UUID):
    """
    Asynchronous risk limit check using real database state.
    """
    async with db_manager.async_session_factory() as db:
        result = await db.execute(
            select(Order, Portfolio)
            .join(Portfolio, Order.portfolio_id == Portfolio.id)
            .where(Order.id == order_id)
        )
        row = result.fetchone()
        if not row:
            return False, "Order not found"
        
        order, portfolio = row
        total_value = (order.limit_price or Decimal("0")) * order.quantity
        
        # Real-world risk logic: Check if order value exceeds 10% of portfolio cash
        if total_value > (portfolio.cash_balance * Decimal("0.10")):
            return False, f"Order value {total_value} exceeds 10% of portfolio cash {portfolio.cash_balance}"
        
        return True, "Success"


@celery_app.task(name="execute_trade_task")
def execute_trade_task(order_id_str: str):
    """
    Production-ready trade execution task.
    Orchestrates order lifecycle from pending to filled/rejected.
    """
    order_id = uuid.UUID(order_id_str)
    
    async def _execute():
        # 1. Risk Check
        is_ok, reason = await check_risk_limits_async(order_id)
        
        async with db_manager.async_session_factory() as db:
            if not is_ok:
                await db.execute(
                    update(Order)
                    .where(Order.id == order_id)
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await db.commit()
                logger.warning("trade_rejected", order_id=order_id_str, reason=reason)
                return {"status": "rejected", "reason": reason}

            # 2. Broker Integration (REAL API)
            broker = get_broker()
            
            # Fetch order details
            res = await db.execute(select(Order).where(Order.id == order_id))
            order = res.scalar_one()
            
            try:
                broker_res = await broker.submit_order(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=order.side,
                    type=order.order_type
                )
                
                # Update with broker-side ID and pending status
                await db.execute(
                    update(Order)
                    .where(Order.id == order_id)
                    .values(
                        status="pending", 
                        broker_order_id=broker_res.get("id"),
                        updated_at=datetime.now(timezone.utc)
                    )
                )
                await db.commit()
                
                logger.info("trade_submitted_to_broker", order_id=order_id_str, broker_id=broker_res.get("id"))
                return {"status": "submitted", "order_id": order_id_str, "broker_id": broker_res.get("id")}
                
            except Exception as e:
                logger.error("broker_submission_failed", order_id=order_id_str, error=str(e))
                await db.execute(
                    update(Order)
                    .where(Order.id == order_id)
                    .values(status="failed", updated_at=datetime.now(timezone.utc))
                )
                await db.commit()
                return {"status": "failed", "error": str(e)}

    loop = asyncio.get_event_loop()
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_execute())
    else:
        return asyncio.run(_execute())


@celery_app.task(name="backtest_strategy_task")
def backtest_strategy_task(strategy_name: str, params: dict):
    """
    High-fidelity backtesting task.
    Fully integrated with the Math Kernel and Historical Data.
    """
    from src.math_kernel.backtesting.kernel import run_simulation_kernel, calculate_metrics_kernel
    
    # 1. Fetch parameters
    initial_capital = params.get("initial_capital", 100000.0)
    symbol = params.get("symbol", "SPY")
    period = params.get("period", "1y")
    
    # FETCH REAL HISTORICAL DATA
    logger.info("fetching_historical_data_for_backtest", symbol=symbol, period=period)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        logger.error("backtest_failed_no_data", symbol=symbol)
        return {"status": "failed", "reason": "No historical data found"}

    option_prices = df["Close"].values.astype(np.float32)
    n_days = len(option_prices)
    
    # Deterministic strategy logic
    target_positions = np.zeros(n_days)
    ma_short = df["Close"].rolling(window=20).mean()
    ma_long = df["Close"].rolling(window=50).mean()
    
    target_positions[ma_short > ma_long] = 1.0
    target_positions[ma_short < ma_long] = -1.0
    target_positions = np.nan_to_num(target_positions)
    
    # 2. Run Kernel
    equity_curve, mtm_pnl, commissions = run_simulation_kernel(
        option_prices,
        target_positions,
        initial_capital
    )
    
    # 3. Calculate Metrics
    total_return, sharpe, sortino, calmar, max_dd = calculate_metrics_kernel(
        equity_curve,
        initial_capital
    )
    
    return {
        "status": "completed",
        "symbol": symbol,
        "metrics": {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd)
        },
        "final_equity": float(equity_curve[-1])
    }