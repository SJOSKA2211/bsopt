"""
Order Engine — Lock-free order life-cycle and risk orchestration.
"""

from __future__ import annotations

import time
import structlog
import numpy as np

from src.shared.shm_mesh import OrderBuffer, ExecutionBuffer
from src.math_kernel.risk_kernels import RiskVectorTracker

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
                idx = i % 1000 # ORDER_BUFFER_CAPACITY
                order = self.orders.view[idx]
                self._execute_order(order, i)
            self._last_head = head

    def _execute_order(self, order_data: np.void, order_id: int):
        """
        Internal execution logic with risk validation.
        """
        symbol = order_data["symbol"].decode("ascii").strip("\x00")
        price = float(order_data["price"])
        qty = int(order_data["quantity"])
        side = int(order_data["side"])
        
        # 1. Pre-Trade Risk Validation
        ok = self.risk.validate_and_update(
            price=price,
            quantity=qty,
            side=side,
            d_delta=float(order_data["delta"]),
            d_gamma=float(order_data["gamma"]),
            d_vega=float(order_data["vega"])
        )
        
        if ok:
            # 2. Simulate Fill (Institutional Grade Mock for HFT workflow)
            # In a live system, this sends to FIX/Binary gateway
            logger.info("order_executed", order_id=order_id, symbol=symbol, price=price)
            self.executions.write_exec(order_id, price, qty, 1) # 1 = FILLED
        else:
            logger.warning("order_risk_veto", order_id=order_id, symbol=symbol)
            self.executions.write_exec(order_id, price, 0, 2) # 2 = REJECTED
