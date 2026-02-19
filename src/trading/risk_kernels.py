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
