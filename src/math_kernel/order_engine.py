from __future__ import annotations
import numpy as np
import structlog
from src.math_kernel.risk_kernels import RiskVectorTracker
from src.shared.shm_mesh import ExecutionBuffer, OrderBuffer

logger = structlog.get_logger(__name__)


class OrderEngine:
    """
    Order Execution Engine.
    Orchestrates pre-trade risk and SHM fill execution.
    """

    def __init__(self, risk_limits: np.ndarray | None = None):
        self.orders = OrderBuffer(create=True)
        self.executions = ExecutionBuffer(create=True)
        self.risk = RiskVectorTracker(limits=risk_limits)
        self._last_head = 0
        logger.info("order_engine_initialized")

    def process_next_orders(self):
        """Poll the SHM OrderBuffer and process any new commands."""
        import struct

        head = struct.unpack_from("q", self.orders.buf, 0)[0]

        if head > self._last_head:
            for i in range(self._last_head, head):
                idx = i % 1000
                order = self.orders.view[idx]
                self._execute_order(order, i)
            self._last_head = head

    def _execute_order(self, order_data: np.void, order_id: int):
        """Internal execution logic with risk validation."""
        symbol = order_data["symbol"].decode("ascii").strip("\x00")
        price = float(order_data["price"])
        qty = int(order_data["quantity"])
        side = int(order_data["side"])

        is_risk_validated = self.risk.validate_and_update(
            price=price,
            quantity=qty,
            side=side,
            d_delta=float(order_data["delta"]),
            d_gamma=float(order_data["gamma"]),
            d_vega=float(order_data["vega"]),
        )

        if not is_risk_validated:
            logger.warning("order_risk_rejected", order_id=order_id, symbol=symbol)
            self.executions.write_exec(order_id, price, 0, 2)
            return

        # Simulate execution
        slippage = 0.0001 * price * (1 if side == 1 else -1)
        fill_price = price + slippage
        fill_qty = qty
        
        # Stochastic Partial Fill (simulation)
        if np.random.rand() > 0.95:
            fill_qty = int(qty * np.random.uniform(0.5, 0.95))

        logger.info(
            "order_executed",
            order_id=order_id,
            symbol=symbol,
            fill_price=round(fill_price, 4),
            fill_qty=fill_qty,
        )

        self.executions.write_exec(order_id, fill_price, fill_qty, 1)