import os
import numpy as np
import struct
from numba import njit
import structlog

from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_ORDER_NAME, ExecutionBuffer, OrderBuffer, RISK_STATE_DTYPE
from src.trading.risk_kernels import _full_risk_check_kernel

logger = structlog.get_logger(__name__)


@njit(cache=True, fastmath=True)
def _order_engine_hot_loop_kernel(
    orders_view: np.ndarray,
    execs_view: np.ndarray,
    risk_state_arr: np.ndarray,
    head_arr: np.ndarray,  # 8-byte array view of orders.buf[:8]
    last_head: int,
    order_id_counter: int,
    max_net_delta: float,
) -> tuple[int, int]:
    """
    GOD-MODE: The absolute hot-path. 
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
        delta = cmd["delta"]
        trade_delta = delta * qty * side

        # 1. Combined Silicon + Portfolio Risk Check
        # Passing risk_state_arr which is a view of SHM RISK_STATE_DTYPE
        ok = _full_risk_check_kernel(
            price, qty, side, trade_delta, risk_state_arr, max_net_delta=max_net_delta
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
            self._risk_state_view = self._risk_shm.view.view(np.float64).reshape(-1) # Flattened float64 view
        except Exception:
            logger.warning("risk_state_shm_missing_falling_back_to_local_only")
            self._risk_state_view = np.array([0.0], dtype=np.float64)

    def run(self, cpu_core: int = 7):
        """Hot loop: Zero-latency order processing via Numba kernel."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("order_engine_pinned", core=cpu_core)
        except Exception:
            pass

        logger.info("order_engine_spinning", shm=SHM_ORDER_NAME)

        while True:
            # CALL THE GOD KERNEL
            new_last_head, new_order_id = _order_engine_hot_loop_kernel(
                self.orders.view,
                self.execs.view,
                self._risk_state_view,
                self._head_arr,
                self._last_head,
                self._order_id_counter,
                self.max_portfolio_delta
            )
            
            if new_last_head > self._last_head:
                # We processed something
                # Note: execs head update is handled by the caller or we can do it here
                # But ExecutionBuffer.write_exec expects us to update its head too.
                # Actually, our Numba kernel just writes the data. We need to update the head of execs buffer.
                # The execution head is the same as the order head in this simple 1:1 model.
                self.execs.buf[:8] = struct.pack("q", new_last_head)
                self._last_head = new_last_head
                self._order_id_counter = new_order_id
            else:
                # Busy-wait
                os.sched_yield()


if __name__ == "__main__":
    engine = OrderEngine()
    engine.run()
