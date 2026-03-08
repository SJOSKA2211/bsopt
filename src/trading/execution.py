import asyncio
import time
from typing import Any

import structlog

from src.blockchain.defi_options import DeFiOptionsProtocol
from src.config import settings
from src.trading.risk_kernels import IncrementalDeltaTracker

logger = structlog.get_logger(__name__)


class OrderExecutor:
    """
    OPTIMIZED: Latency-Optimized Smart Order Executor.
    Handles slippage protection, EIP-1559 gas management, and SOR.
    """

    def __init__(self, protocol: DeFiOptionsProtocol):
        self.protocol = protocol
        self._execution_lock = asyncio.Lock()
        # OPTIMIZED: Incremental tracker to avoid O(N) delta summation
        self._delta_tracker = IncrementalDeltaTracker(max_net_delta=settings.MAX_NET_DELTA)

    async def execute_order(
        self, params: dict[str, Any], max_slippage_pct: float = 0.5
    ) -> dict[str, Any]:
        """Execute a signed transaction with pre-trade risk validation and state persistence."""
        async with self._execution_lock:
            start_time = time.time()
            try:
                # 0. Sync Delta Tracker with Redis (Atomic state management)
                from src.utils.cache import get_redis

                redis = get_redis()
                if redis:
                    cached_delta = await redis.get("portfolio_net_delta")
                    if cached_delta:
                        self._delta_tracker.reset(float(cached_delta))

                # 1. Pre-Trade Risk Validation (Consolidated & Atomic)
                from src.shared.lua_scripts import DISTRIBUTED_RISK_CHECK

                price = float(params.get("price", 0.0))
                quantity = int(params.get("amount", 0))
                side = 1 if params.get("side") == "BUY" else -1
                trade_delta = params.get("delta", 0.0) * quantity * side

                # Atomic Distributed Risk Check
                if redis:
                    try:
                        # Script returns {allowed, new_delta}
                        allowed, new_portfolio_delta = await redis.eval(
                            DISTRIBUTED_RISK_CHECK,
                            1,
                            "portfolio_net_delta",
                            trade_delta,
                            settings.MAX_NET_DELTA,
                        )

                        if not allowed:
                            logger.warning(
                                "order_rejected_delta_risk_lua",
                                params=params,
                                delta=trade_delta,
                                current_total=new_portfolio_delta,
                            )
                            return {"status": "rejected", "reason": "delta_exposure_limit_violation"}

                        # Sync local tracker with atomic state
                        self._delta_tracker.reset(float(new_portfolio_delta))

                    except Exception as e:
                        logger.error("distributed_risk_check_failed", error=str(e))
                        # Fallback to local validation if Redis/LUA fails (better than nothing)
                        if not self._delta_tracker.validate_and_update(trade_delta):
                            return {"status": "rejected", "reason": "delta_exposure_limit_violation"}

                else:
                    # Fallback to local-only if no redis
                    if not self._delta_tracker.validate_and_update(trade_delta):
                        return {"status": "rejected", "reason": "delta_exposure_limit_violation"}

                # 4. Extract params for execution
                contract_address = params.get("contract_address")

                # 5. Check Circuit Breaker
                await self.protocol._check_circuit()

                # 6. Dispatch real transaction
                tx_hash = await self.protocol.buy_option(
                    contract_address=contract_address,
                    amount=quantity,
                    max_slippage=max_slippage_pct / 100.0,
                    params=params,
                )

                duration = (time.time() - start_time) * 1000
                logger.info("order_dispatched_real", tx_hash=tx_hash, latency_ms=duration)

                return {
                    "status": "dispatched",
                    "tx_hash": tx_hash,
                    "latency_ms": duration,
                }

            except Exception as e:
                logger.error("order_execution_failed", error=str(e))
                return {"status": "failed", "error": str(e)}

    async def monitor_transaction(self, tx_hash: str):
        logger.info("monitoring_transaction", tx_hash=tx_hash)
        pass
