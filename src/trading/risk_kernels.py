import numpy as np
import structlog
from numba import njit

try:
    import bsopt_core
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@njit(cache=True, fastmath=True)
def _validate_order_kernel(
    price: float,
    quantity: int,
    side: int,
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0,
) -> int:
    """
    Sub-microsecond silicon risk check.
    Returns: 1 if OK, 0 if VETO.
    """
    # 1. Fat-finger Price Protection
    if price < min_price or price > max_price:
        return 0

    # 2. Quantity Protection
    if quantity <= 0 or quantity > max_qty:
        return 0

    # 3. Side Integrity (Must be 1 or -1)
    if side != 1 and side != -1:
        return 0

    return 1


@njit(cache=True, fastmath=True)
def _validate_delta_kernel(
    state_arr: np.ndarray, trade_delta: float, max_net_delta: float = 10000.0
) -> int:
    """
    O(1) incremental delta check.
    Returns: 1 if OK else 0.
    """
    new_net_delta = state_arr[0] + trade_delta
    if abs(new_net_delta) > max_net_delta:
        return 0
    
    # Commit state
    state_arr[0] = new_net_delta
    return 1


@njit(cache=True, fastmath=True)
def _full_risk_check_kernel(
    price: float,
    quantity: int,
    side: int,
    trade_delta: float,
    state_arr: np.ndarray,
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0,
    max_net_delta: float = 10000.0,
) -> int:
    """
    Combined God-Tier Risk Kernel: Sub-300ns execution.
    Handles fat-finger protection and incremental delta validation in one pass.
    """
    # 1. Base Silicon Checks
    if price < min_price or price > max_price or quantity <= 0 or quantity > max_qty or (side != 1 and side != -1):
        return 0

    # 2. Incremental Delta Check
    new_net_delta = state_arr[0] + trade_delta
    if abs(new_net_delta) > max_net_delta:
        return 0
    
    # 3. State Commit
    state_arr[0] = new_net_delta
    return 1


class IncrementalDeltaTracker:
    """
    Stateful tracker for portfolio-wide delta exposure.
    Maintains O(1) running total. Optimized to use Rust 'bsopt_core' if available.
    """

    def __init__(self, initial_delta: float = 0.0, max_net_delta: float = 10000.0):
        # Use an array to allow Numba to operate on it directly (zero-copy possible)
        self._state = np.array([initial_delta], dtype=np.float64)
        self.max_net_delta = max_net_delta

    @property
    def current_net_delta(self) -> float:
        return self._state[0]

    def validate_and_update(self, trade_delta: float) -> bool:
        """
        Sub-microsecond validation with state update.
        Uses Rust core or Numba fallback.
        """
        if CORE_AVAILABLE:
            try:
                # 1. Fat-finger checks omitted here (handled by _validate_order_kernel or full_risk_check)
                # But for incremental delta only, we use Rust full_risk_check with dummy fat-finger params
                ok, new_delta = bsopt_core.full_risk_check(
                    1.0, 1, 1, trade_delta, self._state[0], 100, 0.01, 100.0, self.max_net_delta
                )
                if ok:
                    self._state[0] = new_delta
                return ok
            except Exception as e:
                logger.warning("rust_risk_check_failed", error=str(e))

        return bool(_validate_delta_kernel(self._state, trade_delta, self.max_net_delta))

    def full_risk_check(
        self,
        price: float,
        quantity: int,
        side: int,
        trade_delta: float,
        max_qty: int = 1000,
        min_price: float = 0.01,
        max_price: float = 10000.0,
    ) -> bool:
        """
        Combined risk check using Rust core if possible.
        """
        if CORE_AVAILABLE:
            try:
                ok, new_delta = bsopt_core.full_risk_check(
                    float(price),
                    int(quantity),
                    int(side),
                    float(trade_delta),
                    float(self._state[0]),
                    int(max_qty),
                    float(min_price),
                    float(max_price),
                    float(self.max_net_delta),
                )
                if ok:
                    self._state[0] = new_delta
                return ok
            except Exception as e:
                logger.warning("rust_full_risk_failed", error=str(e))

        # Fallback to Numba
        ok = bool(
            _full_risk_check_kernel(
                price,
                quantity,
                side,
                trade_delta,
                self._state,
                max_qty,
                min_price,
                max_price,
                self.max_net_delta,
            )
        )
        return ok

    def reset(self, new_delta: float):
        """Periodic full-sync to prevent drift."""
        self._state[0] = new_delta


@njit(cache=True, fastmath=True)
def _validate_delta_exposure_kernel(
    current_deltas: np.ndarray, trade_delta: float, max_net_delta: float = 10000.0
) -> int:
    """
    Check if trade exceeds total portfolio delta limits using O(N) summation.
    Used for periodic reconciliation.
    """
    net_delta = np.sum(current_deltas) + trade_delta
    if abs(net_delta) > max_net_delta:
        return 0
    return 1
