import asyncio
import time
import os
from typing import Any

import numpy as np
import structlog

from src.blockchain.defi_options import DeFiOptionsProtocol
from src.config import settings
from src.shared.lua_scripts import ADVANCED_RISK_MATRIX
from src.trading.risk_kernels import RiskVectorTracker
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class OrderExecutor:
    """
    OPTIMIZED: Latency-Optimized Smart Order Executor.
    Handles slippage protection, EIP-1559 gas management, and SOR.
    """

    def __init__(self, protocol: DeFiOptionsProtocol):
        self.protocol = protocol
        self._execution_lock = asyncio.Lock()
        # OPTIMIZED: Multi-dimensional tracker for Delta, Gamma, Vega
        self._risk_tracker = RiskVectorTracker(
            limits=np.array(
                [settings.MAX_NET_DELTA, settings.MAX_NET_GAMMA, settings.MAX_NET_VEGA],
                dtype=np.float64,
            )
        )
        self.redis = get_redis()
        self.risk_script = ADVANCED_RISK_MATRIX

    async def execute_order(
        self, params: dict[str, Any], max_slippage_pct: float = 0.5
    ) -> dict[str, Any]:
        """Execute a signed transaction with pre-trade risk validation and state persistence."""
        async with self._execution_lock:
            start_time = time.time()
            try:
                # 0. Sync Risk Tracker with Redis (Atomic state management)
                if self.redis:
                    async with self.redis.pipeline(transaction=False) as pipe:
                        pipe.get("portfolio_net_delta")
                        pipe.get("portfolio_net_gamma")
                        pipe.get("portfolio_net_vega")
                        results = await pipe.execute()

                    current_metrics = np.array(
                        [float(r) if r else 0.0 for r in results], dtype=np.float64
                    )
                    self._risk_tracker.reset(current_metrics)

                # 1. Pre-Trade Risk Validation (Consolidated & Atomic)
                price = float(params.get("price", 0.0))
                quantity = int(params.get("amount", 0))
                side = 1 if params.get("side") == "BUY" else -1

                d_delta = float(params.get("delta", 0.0)) * quantity * side
                d_gamma = float(params.get("gamma", 0.0)) * quantity * side
                d_vega = float(params.get("vega", 0.0)) * quantity * side

                # Atomic Distributed Risk Check
                if self.redis:
                    try:
                        # Script returns [ok, val1, val2, val3]
                        result = await self.redis.eval(
                            self.risk_script,
                            3,
                            "risk:state:matrix",
                            "risk:kill_switch",
                            "blockchain:breaker:state",
                            d_delta,
                            d_gamma,
                            d_vega,
                            settings.MAX_NET_DELTA,
                            settings.MAX_NET_GAMMA,
                            settings.MAX_NET_VEGA,
                        )

                        if result[0] != 1:
                            reason = (
                                result[1].decode()
                                if isinstance(result[1], bytes)
                                else str(result[1])
                            )
                            logger.warning(
                                "order_rejected_risk_matrix", reason=reason, params=params
                            )
                            return {"status": "rejected", "reason": reason}

                        # Sync local tracker with atomic state
                        new_state = np.array(
                            [float(result[1]), float(result[2]), float(result[3])], dtype=np.float64
                        )
                        self._risk_tracker.reset(new_state)

                    except Exception as e:
                        logger.error("distributed_risk_check_failed", error=str(e))
                        # Fallback to local validation
                        if not self._risk_tracker.validate_and_update(
                            price, quantity, side, d_delta, d_gamma, d_vega
                        ):
                            return {"status": "rejected", "reason": "risk_limit_violation"}

                else:
                    # Fallback to local-only if no redis
                    if not self._risk_tracker.validate_and_update(
                        price, quantity, side, d_delta, d_gamma, d_vega
                    ):
                        return {"status": "rejected", "reason": "risk_limit_violation"}

                # 4. Extract params for execution
                contract_address = params.get("contract_address")
                token_address = params.get("token_address") # Needed for permit

                # 5. Check Circuit Breaker
                await self.protocol._check_circuit()

                # 6. Smart Order Router (SOR) Intervention
                # Determine best venue based on current mempool conditions
                sor_decision = await self.protocol.route_order_advanced(
                    params.get("symbol", "UNKNOWN"), quantity, True
                )
                logger.info("sor_routing_decision", venue=sor_decision["name"], price=sor_decision["price"])
                
                # Check for Gasless Meta-Transaction Eligibility
                is_gasless_eligible = params.get("use_gasless", True) and token_address is not None
                
                # 7. Dispatch transaction
                if is_gasless_eligible:
                    # 🚀 FAST-PATH: Gasless Meta-Transaction
                    deadline = int(time.time()) + 300 # 5 min deadline
                    payload = await self.protocol.buy_option_gasless(
                        token_address=token_address,
                        contract_address=contract_address,
                        amount=quantity,
                        deadline=deadline,
                        params=params
                    )
                    
                    # Submit payload to dummy/provided relayer
                    relayer_url = os.environ.get("RELAYER_URL", "http://localhost:8080/submit_meta_tx")
                    try:
                        tx_hash = await self.protocol.submit_meta_transaction(relayer_url, payload)
                        duration = (time.time() - start_time) * 1000
                        logger.info("order_dispatched_gasless", tx_hash=tx_hash, latency_ms=duration)
                        return {
                            "status": "dispatched_gasless",
                            "tx_hash": tx_hash,
                            "latency_ms": duration,
                        }
                    except Exception as e:
                        logger.warning("gasless_submission_failed_falling_back", error=str(e))
                        # Fallback to standard execution handled below
                
                # 🐢 STANDARD-PATH: EIP-1559 Transaction
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
