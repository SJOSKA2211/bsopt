import numpy as np
from numba import float32, int32, njit


@njit(
    float32[:, :](
        float32,
        float32,
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32[:],
        float32[:, :],
        int32,
        int32,
    ),
    cache=True,
    fastmath=True,
    error_model="numpy",
)
def _fused_state_kernel(
    balance,
    initial_balance,
    positions,
    prices,
    strikes,
    greeks,
    indicators,
    window_buffer,
    window_idx,
    window_size,
):
    """
    Fused kernel for state matrix construction (2D) with Spectral Features.
    Returns (window_size, 100) instead of flattened vector.
    """
    # Create the current state vector (100 dims)
    state = np.zeros(100, dtype=np.float32)

    # 1. Portfolio (11 dims)
    state[0] = balance / initial_balance
    for i in range(10):
        state[1 + i] = positions[i]

    # 2. Market (10 dims) - Log-Moneyness + Fourier Base
    for i in range(10):
        p = max(prices[i], 1e-6)
        k = max(strikes[i], 1e-6)
        log_m = np.log(p / k)
        state[11 + i] = log_m

    # 3. Greeks (40 dims) - Tanh Scaling
    # Reduced from 50 to 40 to make room for Fourier features
    for i in range(40):
        state[21 + i] = np.tanh(greeks[i])

    # 4. Spectral Features (10 dims) - Multi-scale Fourier Base
    # Captures cyclical micro-structure at log-spaced frequencies.
    for i in range(5):
        p_norm = prices[i] / 100.0
        # Multi-scale frequencies (1.0, 2.0, 4.0, 8.0, 16.0)
        freq = 2.0**i
        state[61 + i] = np.tanh(np.sin(p_norm * np.pi * freq))
        state[66 + i] = np.tanh(np.cos(p_norm * np.pi * freq))

    # 5. Indicators (20 dims)
    # Adjusted indices to start from 71 (stays the same as before)
    for i in range(20):
        state[71 + i] = indicators[i]

    # 6. Temporal Stacking (Circular Buffer)
    # Write to window buffer at current index
    idx = window_idx % window_size
    window_buffer[idx] = state

    # Return 2D window (ordered chronologically or latest-first)
    # Let's go with Chronological for the Transformer (oldest to newest)
    out = np.zeros((window_size, 100), dtype=np.float32)
    for i in range(window_size):
        # Calculate index for chronologically ordered window (i steps ago from current)
        # i=0 is oldest, i=window_size-1 is newest
        src_idx = (window_idx - (window_size - 1 - i)) % window_size
        out[i] = window_buffer[src_idx]

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


@njit(cache=True, fastmath=True)
def _trading_step_kernel(
    action,
    current_prices,
    current_positions,
    current_balance,
    transaction_cost_pct,
    initial_balance
):
    """
    God-Tier: Full Environment Step Fusion.
    Executes rebalancing, cost calculation, and state evolution in machine code.
    Returns: (new_positions, new_balance, new_portfolio_value, reward)
    """
    p_safe = np.zeros(10, dtype=np.float32)
    for i in range(10):
        p_safe[i] = max(current_prices[i], 1e-6)

    # 1. Current Valuation
    current_asset_val = 0.0
    for i in range(10):
        current_asset_val += current_positions[i] * p_safe[i]
    portfolio_value = current_balance + current_asset_val

    # 2. Target units based on action weights
    target_units = np.zeros(10, dtype=np.float32)
    trades = np.zeros(10, dtype=np.float32)
    total_transaction_costs = 0.0
    total_asset_costs = 0.0

    for i in range(10):
        # Action is expected to be in [-1, 1] range (weights)
        target_units[i] = (action[i] * portfolio_value) / (p_safe[i] + 1e-9)
        trades[i] = target_units[i] - current_positions[i]
        
        # Costs
        total_transaction_costs += abs(trades[i]) * p_safe[i] * transaction_cost_pct
        total_asset_costs += trades[i] * p_safe[i]

    # 3. State Update
    new_balance = current_balance - (total_transaction_costs + total_asset_costs)
    new_positions = target_units

    # 4. Valuation after price move (assuming prices here are current at end of step)
    new_asset_val = 0.0
    for i in range(10):
        new_asset_val += new_positions[i] * p_safe[i]
    new_portfolio_value = new_balance + new_asset_val

    # 5. Reward Calculation (O(1) return)
    reward = (new_portfolio_value - portfolio_value) / max(portfolio_value, 1e-6)

    # 6. Stability Penalties (Hardcoded for speed)
    # Drawdown penalty
    dd = (initial_balance - new_portfolio_value) / initial_balance
    if dd > 0.1:
        reward -= 0.05 * (dd - 0.1)

    return new_positions, new_balance, new_portfolio_value, reward
