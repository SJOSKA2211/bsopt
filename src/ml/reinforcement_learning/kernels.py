import numpy as np
import structlog
from numba import njit

logger = structlog.get_logger()

#  SILICON KERNELS: JIT-fused state construction
# Targets AVX-512 and aggressive unrolling.

@njit(cache=True, fastmath=True, error_model='numpy')
def _fused_state_kernel(
    balance, initial_balance, 
    positions, prices, strikes, 
    greeks, indicators,
    window_buffer, window_idx, window_size
):
    """
    Fused kernel for state vector construction.
    Fuses scaling, log-moneyness, and tanh-normalization.
    """
    state = np.zeros(100, dtype=np.float32)
    
    # 1. Portfolio (11 dims)
    state[0] = balance / initial_balance
    for i in range(10):
        state[1+i] = positions[i]
        
    # 2. Market (10 dims) - Log-Moneyness
    for i in range(10):
        p = max(prices[i], 1e-6)
        k = max(strikes[i], 1e-6)
        state[11+i] = np.log(p / k)
        
    # 3. Greeks (50 dims) - Tanh Scaling
    for i in range(50):
        state[21+i] = np.tanh(greeks[i])
        
    # 4. Indicators (20 dims)
    for i in range(20):
        state[71+i] = indicators[i]
        
    # 5. Temporal Stacking (Circular Buffer)
    # Write to window buffer at current index
    idx = window_idx % window_size
    window_buffer[idx] = state
    
    # Return flattened window (latest first)
    # This involves a copy, but it's a contiguous block copy.
    out = np.zeros(100 * window_size, dtype=np.float32)
    for i in range(window_size):
        src_idx = (window_idx - i) % window_size
        start = i * 100
        out[start:start+100] = window_buffer[src_idx]
        
    return out

@njit(cache=True, fastmath=True)
def _calculate_reward_kernel(positions, current_prices, prev_portfolio_value, balance):
    """Zero-allocation reward calculation."""
    option_val = 0.0
    for i in range(10):
        option_val += positions[i] * current_prices[i]
    
    current_val = balance + option_val
    ret = (current_val - prev_portfolio_value) / max(prev_portfolio_value, 1e-6)
    return current_val, ret
