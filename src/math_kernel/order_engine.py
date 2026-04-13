from __future__ import annotations

import numpy as np
import structlog

from src.math_kernel.risk_kernels import RiskVectorTracker
from src.shared.shm_mesh import ExecutionBuffer, OrderBuffer

logger = structlog.get_logger(__name__)


class OrderEngine:
    """
    Divine-level Order Execution Engine.
    Orchestrates pre-trade risk and SHM fill execution.
    """

    def __init__(self, risk_limits: np.ndarray | None = None):
        self.orders = OrderBuffer(create=True)
        self.executions = ExecutionBuffer(create=True)
        self.risk = RiskVectorTracker(limits=risk_limits)
        self._last_head = 0
        logger.info("order_engine_ready", status="listening_shm")

    def process_next_orders(self):
        """
        Poll the SHM OrderBuffer and process any new commands.
        """
        import struct

        head = struct.unpack_from("q", self.orders.buf, 0)[0]

        if head > self._last_head:
            for i in range(self._last_head, head):
                idx = i % 1000  # ORDER_BUFFER_CAPACITY
                order = self.orders.view[idx]
                self._execute_order(order, i)
            self._last_head = head

    def _execute_order(self, order_data: np.void, order_id: int):
        """
        Internal execution logic with risk validation and HFT simulation.
        """
        symbol = order_data["symbol"].decode("ascii").strip("\x00")
        price = float(order_data["price"])
        qty = int(order_data["quantity"])
        side = int(order_data["side"])

        # 1. PRE-TRADE RISK VALIDATION (Divine Veto)
        ok = self.risk.validate_and_update(
            price=price,
            quantity=qty,
            side=side,
            d_delta=float(order_data["delta"]),
            d_gamma=float(order_data["gamma"]),
            d_vega=float(order_data["vega"]),
        )

        if ok:
            # Simulate micro-slippage (0.1 bps) and market impact
            slippage = 0.0001 * price * (1 if side == 1 else -1)
            fill_price = price + slippage

            # Stochastic Partial Fill (95% full fill, 5% partial)
            fill_qty = qty
            if np.random.rand() > 0.95:
                fill_qty = int(qty * np.random.uniform(0.5, 0.95))

            # HFT Latency Simulation (Local Clock)
            # In a live system, this sends to FIX/Binary gateway
            logger.info(
                "order_executed_sim",
                order_id=order_id,
                symbol=symbol,
                fill_price=round(fill_price, 4),
                fill_qty=fill_qty,
                status="FILLED" if fill_qty == qty else "PARTIAL",
            )

            self.executions.write_exec(order_id, fill_price, fill_qty, 1)  # 1 = FILLED/PARTIAL
        else:
            logger.warning(
                "order_risk_veto", order_id=order_id, symbol=symbol, reason="LIMIT_BREACH"
            )
            self.executions.write_exec(order_id, price, 0, 2)  # 2 = REJECTED