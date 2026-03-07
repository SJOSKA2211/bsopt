import os

import structlog

from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_ORDER_NAME, ExecutionBuffer, OrderBuffer

logger = structlog.get_logger(__name__)


class OrderEngine:
    """
    The Nerve Impulse: High-Frequency Order Entry Gateway.
    OPTIMIZED: Zero-copy polling and zero-allocation spin-path.
    """

    def __init__(self, max_portfolio_delta: float = 10000.0):
        tune_gc()
        self.orders = OrderBuffer(create=False)
        self.execs = ExecutionBuffer(create=False)
        self._last_head = 0
        self._order_id_counter = 1000

        # Pre-bind for hot loop speed
        import struct

        self._struct_q = struct.Struct("q")
        self._head_mv = self.orders.buf[:8]

        from src.trading.risk_kernels import IncrementalDeltaTracker, _validate_order_kernel

        self._risk_check = _validate_order_kernel
        # 🛡️ SOLENYA SHIELD: Incremental Portfolio Risk Tracker
        self._delta_tracker = IncrementalDeltaTracker(max_net_delta=max_portfolio_delta)
        self._drift_counter = 0

    def run(self, cpu_core: int = 7):
        """Hot loop: Zero-latency order processing."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("order_engine_pinned", core=cpu_core)
        except Exception:
            pass

        logger.info("order_engine_spinning", shm=SHM_ORDER_NAME)

        while True:
            # OPTIMIZED: Zero-copy head poll
            current_head = self._struct_q.unpack(self._head_mv)[0]

            if current_head > self._last_head:
                # New order command!
                cmd = self.orders.view[self._last_head % 1000]
                price = float(cmd["price"])
                qty = int(cmd["quantity"])
                side = int(cmd["side"])
                trade_delta = float(cmd["delta"]) * qty * side

                # 1. Base Silicon Risk Check (Price/Qty/Side)
                if self._risk_check(price, qty, side):
                    # 2. Portfolio-Level Incremental Delta Check
                    if self._delta_tracker.validate_and_update(trade_delta):
                        #  BINARY FIRE
                        order_id = self._order_id_counter
                        self._order_id_counter += 1

                        self.execs.write_exec(order_id, price, qty, 1)
                    else:
                        logger.warning(
                            "risk_veto_delta_limit_exceeded",
                            delta=self._delta_tracker.current_net_delta,
                            trade_delta=trade_delta,
                        )
                        self.execs.write_exec(-1, price, qty, 0)
                else:
                    self.execs.write_exec(-1, price, qty, 0)

                self._last_head += 1
                self._drift_counter += 1

                # Periodic drift correction (Placeholder for real DB sync)
                if self._drift_counter >= 1000:
                    self._sync_delta()
                    self._drift_counter = 0
            else:
                # Busy-wait with pause or yield
                os.sched_yield()

    def _sync_delta(self):
        """Sync tracker with source of truth to prevent numerical drift."""
        # TODO: Pull actual net delta from database or SHM portfolio view
        logger.debug("delta_tracker_sync_check", current=self._delta_tracker.current_net_delta)


if __name__ == "__main__":
    engine = OrderEngine()
    engine.run()
