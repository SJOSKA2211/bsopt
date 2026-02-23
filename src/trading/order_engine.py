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

    def __init__(self):
        tune_gc()
        self.orders = OrderBuffer(create=False)
        self.execs = ExecutionBuffer(create=False)
        self._last_head = 0
        self._order_id_counter = 1000

        # Pre-bind for hot loop speed
        import struct

        self._struct_q = struct.Struct("q")
        self._head_mv = self.orders.buf[:8]

        from src.trading.risk_kernels import _validate_order_kernel

        self._risk_check = _validate_order_kernel

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

                # 🛡️ SOLENYA SHIELD: Direct JIT call
                if self._risk_check(
                    float(cmd["price"]), int(cmd["quantity"]), int(cmd["side"])
                ):
                    #  BINARY FIRE
                    order_id = self._order_id_counter
                    self._order_id_counter += 1

                    self.execs.write_exec(
                        order_id, float(cmd["price"]), int(cmd["quantity"]), 1
                    )
                else:
                    self.execs.write_exec(
                        -1, float(cmd["price"]), int(cmd["quantity"]), 0
                    )

                self._last_head += 1
            else:
                # Busy-wait with pause or yield
                os.sched_yield()


if __name__ == "__main__":
    engine = OrderEngine()
    engine.run()
