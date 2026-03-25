import numpy as np

# High-Performance Indicator Kernels (Pure NumPy)
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _ema_kernel(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    # Find first non-NaN
    start_idx = -1
    for i in range(n):
        if not np.isnan(values[i]):
            start_idx = i
            break

    if start_idx == -1:
        return out

    out[start_idx] = values[start_idx]
    for i in range(start_idx + 1, n):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out

def get_ema(values: np.ndarray, span: int) -> np.ndarray:
    return _ema_kernel(values, span)

@njit(cache=True, fastmath=True)
def _rsi_kernel(prices: np.ndarray, length: int) -> np.ndarray:
    n = prices.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= length:
        return out

    # Calculate initial average gain/loss
    gains = 0.0
    losses = 0.0
    for i in range(1, length + 1):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff

    avg_gain = gains / length
    avg_loss = losses / length

    if avg_loss == 0:
        out[length] = 100.0
    else:
        out[length] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    # Wilder's Smoothing
    for i in range(length + 1, n):
        diff = prices[i] - prices[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0

        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length

        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out

def get_rsi(prices: np.ndarray, length: int = 14) -> np.ndarray:
    return _rsi_kernel(prices, length)

def get_bbands(
    prices: np.ndarray, length: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OPTIMIZED: O(n) Bollinger Bands using rolling windows via cumulative sums."""
    # Using pandas rolling for convenience if available, but staying in NumPy for pure JIT-level speed
    # We use the relationship: Var(X) = E[X^2] - (E[X])^2
    n = len(prices)
    if n < length:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    # 1. Moving Average (E[X])
    cumsum = np.cumsum(prices)
    mid = np.full(n, np.nan)
    mid[length - 1 :] = (
        cumsum[length - 1 :] - np.concatenate([np.zeros(1), cumsum[: n - length]])
    ) / length

    # 2. Moving Variance
    cumsum_sq = np.cumsum(prices**2)
    mean_sq = (
        cumsum_sq[length - 1 :] - np.concatenate([np.zeros(1), cumsum_sq[: n - length]])
    ) / length
    variance = np.maximum(mean_sq - mid[length - 1 :] ** 2, 0)
    std = np.sqrt(variance)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper[length - 1 :] = mid[length - 1 :] + num_std * std
    lower[length - 1 :] = mid[length - 1 :] - num_std * std

    return lower, mid, upper

def get_macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD."""
    fast_ema = get_ema(prices, fast)
    slow_ema = get_ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = get_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

@njit(cache=True, fastmath=True)
def _atr_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    n = high.shape[0]
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return _ema_kernel(tr, length)

def get_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """Average True Range."""
    return _atr_kernel(high, low, close, length)

@njit(cache=True, fastmath=True, parallel=True)
def _adx_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    n = high.shape[0]
    up_move = np.zeros(n, dtype=np.float64)
    down_move = np.zeros(n, dtype=np.float64)

    for i in prange(1, n):
        up_move[i] = high[i] - high[i - 1]
        down_move[i] = low[i - 1] - low[i]

    pos_dm = np.zeros(n, dtype=np.float64)
    neg_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in prange(1, n):
        if up_move[i] > down_move[i] and up_move[i] > 0:
            pos_dm[i] = up_move[i]
        if down_move[i] > up_move[i] and down_move[i] > 0:
            neg_dm[i] = down_move[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    smooth_tr = _ema_kernel(tr, length)
    smooth_pos_dm = _ema_kernel(pos_dm, length)
    smooth_neg_dm = _ema_kernel(neg_dm, length)

    adx = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        tr_val = smooth_tr[i] if smooth_tr[i] != 0 else 1e-12
        pos_di = 100.0 * smooth_pos_dm[i] / tr_val
        neg_di = 100.0 * smooth_neg_dm[i] / tr_val

        di_sum = pos_di + neg_di
        di_sum = di_sum if di_sum != 0 else 1e-12
        dx = 100.0 * abs(pos_di - neg_di) / di_sum
        adx[i] = dx

    return _ema_kernel(adx, length)

def get_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """Average Directional Index."""
    return _adx_kernel(high, low, close, length)
