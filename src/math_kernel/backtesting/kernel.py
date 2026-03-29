import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def run_simulation_kernel(
    option_prices: np.ndarray,
    target_positions: np.ndarray,
    initial_capital: float,
    transaction_cost_pct: float = 0.001,
):
    """
    Weaponized backtesting kernel using Numba.
    Operates on raw NumPy arrays for maximum performance.
    """
    n = len(option_prices)
    equity_curve = np.zeros(n)
    mtm_pnl = np.zeros(n)
    commissions = np.zeros(n)

    current_equity = initial_capital
    equity_curve[0] = initial_capital

    prev_pos = 0.0

    for i in range(1, n):
        # 1. Calculate Trades
        pos = target_positions[i]
        trade_size = pos - prev_pos

        # 2. Commissions
        comm = abs(trade_size * option_prices[i] * transaction_cost_pct)
        commissions[i] = comm

        # 3. P&L (Mark-to-Market)
        price_change = option_prices[i] - option_prices[i - 1]
        pnl = (prev_pos * price_change) - comm
        mtm_pnl[i] = pnl

        # 4. Update Equity
        current_equity += pnl
        equity_curve[i] = current_equity

        # 5. Shift state
        prev_pos = pos

    return equity_curve, mtm_pnl, commissions

@njit(cache=True, fastmath=True)
def calculate_metrics_kernel(equity_curve: np.ndarray, initial_capital: float):
    """Annualized metrics calculation in the kernel."""
    returns = np.diff(equity_curve) / equity_curve[:-1]

    # Standard Metrics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # Sortino Ratio (Downside-only standard deviation)
    downside_returns = returns[returns < 0.0]
    std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    sortino = (mean_ret / std_downside) * np.sqrt(252) if std_downside > 0 else 0.0

    # Max Drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for i in range(1, len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (equity_curve[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Calmar Ratio (Annualized Return / Max Drawdown)
    total_return = (equity_curve[-1] / initial_capital) - 1.0
    calmar = (total_return / abs(max_dd)) if max_dd < 0 else 0.0

    return total_return, sharpe, sortino, calmar, max_dd
