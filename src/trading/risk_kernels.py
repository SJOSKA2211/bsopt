import numpy as np
from numba import njit


@njit(cache=True)
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

    # 3. Side Integrity
    if side != 1 and side != -1:
        return 0

    return 1


class IncrementalDeltaTracker:
    """
    Stateful tracker for portfolio-wide delta exposure.
    Maintains O(1) running total for sub-microsecond validation.
    """

    def __init__(self, initial_delta: float = 0.0, max_net_delta: float = 10000.0):
        self.current_net_delta = initial_delta
        self.max_net_delta = max_net_delta

    def validate_and_update(self, trade_delta: float) -> bool:
        """
        Sub-microsecond validation with state update.
        """
        ok, new_delta = _validate_incremental_delta_kernel(
            self.current_net_delta, trade_delta, self.max_net_delta
        )
        if ok:
            self.current_net_delta = new_delta
            return True
        return False

    def reset(self, new_delta: float):
        """Periodic full-sync to prevent drift."""
        self.current_net_delta = new_delta


@njit(cache=True)
def _validate_incremental_delta_kernel(
    current_net_delta: float, trade_delta: float, max_net_delta: float = 10000.0
) -> tuple[int, float]:
    """
    O(1) incremental delta check.
    Returns: (1 if OK else 0, new_net_delta)
    """
    new_net_delta = current_net_delta + trade_delta
    if abs(new_net_delta) > max_net_delta:
        return 0, current_net_delta
    return 1, new_net_delta


@njit(cache=True)
def _validate_delta_exposure_kernel(
    current_deltas: np.ndarray, trade_delta: float, max_net_delta: float = 10000.0
) -> int:
    """
    Check if trade exceeds total portfolio delta limits.
    """
    net_delta = np.sum(current_deltas) + trade_delta
    if abs(net_delta) > max_net_delta:
        return 0
    return 1
