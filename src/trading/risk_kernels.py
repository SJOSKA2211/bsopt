
import numpy as np
from numba import njit

@njit(cache=True)
def _validate_order_kernel(
    price: float, 
    quantity: int, 
    side: int,
    max_qty: int = 1000,
    min_price: float = 0.01,
    max_price: float = 10000.0
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
def _validate_exposure_kernel(
    current_positions: np.ndarray,
    target_trade: float,
    index: int,
    max_position: float = 5000.0
) -> int:
    """
    Check if trade exceeds position limits for a symbol.
    """
    new_pos = current_positions[index] + target_trade
    if abs(new_pos) > max_position:
        return 0
    return 1
