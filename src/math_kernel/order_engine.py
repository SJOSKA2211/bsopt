import ctypes
import os

import numpy as np
import structlog
from numba import njit

from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_ORDER_NAME, ExecutionBuffer, OrderBuffer
from src.math_kernel.risk_kernels import _full_risk_check_v2_kernel

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


logger = structlog.get_logger(__name__)


def _get_addr(buf):
    """Zero-overhead address resolution for SharedMemory buffers."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


@njit(cache=True, fastmath=True)
def _order_engine_hot_loop_kernel(
    orders_view: np.ndarray,
    execs_view: np.ndarray,
    risk_state_arr: np.ndarray,  # 0:d, 1:g, 2:v, 3:max_d, 4:max_g, 5:max_v
    head_arr: np.ndarray,  # 8-byte array view of orders.buf[:8]
    last_head: int,
    order_id_counter: int,
) -> tuple[int, int]:
    """
    HIGH-PERFORMANCE: The absolute hot-path.
    Processes all pending orders in a single JIT-compiled pass.
    """
    current_head = head_arr[0]
    processed_count = 0

    while last_head < current_head:
        # OPTIMIZED: Direct NumPy access instead of Python dict/msgspec
        idx = last_head % 1000
        cmd = orders_view[idx]

        price = cmd["price"]
        qty = cmd["quantity"]
        side = cmd["side"]

        # Incremental Greeks for this trade
        d_delta = cmd["delta"] * qty * side
        d_gamma = cmd["gamma"] * qty * side
        d_vega = cmd["vega"] * qty * side

        # 1. Combined Silicon + Portfolio Risk Check (Greeks Matrix)
        # Using expanded risk_state_arr for both current state and limits
        ok = _full_risk_check_v2_kernel(
            price,
            qty,
            side,
            d_delta,
            d_gamma,
            d_vega,
            risk_state_arr[0:3],
            risk_state_arr[3:6],
        )

        # 2. Execution Response
        exec_idx = last_head % 1000
        if ok:
            order_id = order_id_counter
            order_id_counter += 1
            # status=1 for success
            execs_view[exec_idx]["order_id"] = order_id
            execs_view[exec_idx]["status"] = 1
        else:
            # status=0 for reject
            execs_view[exec_idx]["order_id"] = -1
            execs_view[exec_idx]["status"] = 0

        execs_view[exec_idx]["fill_price"] = price
        execs_view[exec_idx]["fill_qty"] = qty
        # exec_ts_ns would go here if we wanted to log it in Numba (requires time() call)

        last_head += 1
        processed_count += 1

    return last_head, order_id_counter


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
        self.max_portfolio_delta = max_portfolio_delta

        # Map the head index directly as a numpy array for Numba
        self._head_arr = np.frombuffer(self.orders.buf, dtype=np.int64, count=1)

        # Initialize High-Speed Risk State Buffer
        try:
            from src.shared.shm_mesh import RiskStateBuffer

            self._risk_shm = RiskStateBuffer(create=False)
            # This is a view of the current_delta, max_delta, last_sync_ts_ns
            # We want just the current_delta field for the risk kernel
            self._risk_state_view = self._risk_shm.view.view(np.float64).reshape(
                -1
            )  # Flattened float64 view
        except Exception:
            logger.warning("risk_state_shm_missing_falling_back_to_local_only")
            self._risk_state_view = np.array([0.0], dtype=np.float64)

    def run(self, cpu_core: int = 7):
        """Hot loop: Zero-latency order processing via Rust or Numba kernel."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("order_engine_pinned", src.shared=cpu_core)
        except Exception:
            pass

        logger.info("order_engine_spinning", shm=SHM_ORDER_NAME, core_available=CORE_AVAILABLE)

        # Get raw pointers for Rust src.shared if available
        orders_ptr = 0
        execs_ptr = 0
        risk_ptr = 0

        if CORE_AVAILABLE:
            orders_ptr = _get_addr(self.orders.buf)
            execs_ptr = _get_addr(self.execs.buf)
            risk_ptr = _get_addr(self._risk_state_view.data)

        while True:
            if CORE_AVAILABLE:
                # CALL THE RUST GOD KERNEL (Zero-Allocation, Zero-GIL)
                self._last_head, self._order_id_counter = bsopt_core.order_engine_loop(
                    orders_ptr,
                    execs_ptr,
                    risk_ptr,
                    self._last_head,
                    self._order_id_counter,
                    float(self.max_portfolio_delta),
                    1000,  # max_qty
                )
            else:
                # Fallback to Numba kernel
                new_last_head, new_order_id = _order_engine_hot_loop_kernel(
                    self.orders.view,
                    self.execs.view,
                    self._risk_state_view,
                    self._head_arr,
                    self._last_head,
                    self._order_id_counter,
                )

                if new_last_head > self._last_head:
                    # OPTIMIZED: Update exec head via direct array view (O(1))
                    # execs.buf[:8] is the head index
                    np.frombuffer(self.execs.buf, dtype=np.int64, count=1)[0] = new_last_head
                    self._last_head = new_last_head
                    self._order_id_counter = new_order_id

            # Yield to OS to keep things responsive if no work
            os.sched_yield()


if __name__ == "__main__":
    engine = OrderEngine()
    engine.run()
