import numpy as np

# 🚀 Rick Optimization: Robust Numba & CUDA detection
HAS_NUMBA = True
HAS_CUDA = False
try:
    from numba import cuda, jit, njit, prange
    try:
        HAS_CUDA = cuda.is_available()
    except Exception:
        HAS_CUDA = False
except ImportError:
    HAS_NUMBA = False
    HAS_CUDA = False
    # Mocking decorators for Jerry-environments without Numba
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    class CudaMock:
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    cuda = CudaMock()

# =============================================================================
# High-Performance Indicator Kernels (Numba NJIT)
# =============================================================================

@jit(nopython=True, cache=True)
def _numba_ema(values: np.ndarray, span: int) -> np.ndarray:
    """Carr-Madan compliant EMA via recurrence relation. O(N) complexity."""
    alpha = 2 / (span + 1)
    out = np.empty_like(values)
    out[:] = np.nan
    
    # Identify first finite sequence entry
    first_valid_idx = -1
    for i in range(len(values)):
        if not np.isnan(values[i]):
            first_valid_idx = i
            break
            
    if first_valid_idx == -1:
        return out
        
    out[first_valid_idx] = values[first_valid_idx]
    for i in range(first_valid_idx + 1, len(values)):
        if np.isnan(values[i]):
            out[i] = out[i-1] # Recurrence propagation through discontinuities
        else:
            out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
            
    return out

@jit(nopython=True, cache=True)
def _numba_rsi(prices: np.ndarray, length: int = 14) -> np.ndarray:
    """Wilder's RSI implementation. NJIT parallel-capable but serial for small windows."""
    out = np.full_like(prices, np.nan)
    if len(prices) <= length:
        return out
        
    deltas = prices[1:] - prices[:-1]
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # SMA Seed for Wilder's Smoothing
    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])
    
    if avg_loss == 0:
        out[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[length] = 100.0 - (100.0 / (1.0 + rs))
        
    # Recurrent update loop
    for i in range(length + 1, len(prices)):
        delta = prices[i] - prices[i-1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return out

@jit(nopython=True, cache=True)
def _numba_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Vectorized MACD oscillator decomposition."""
    ema_fast = _numba_ema(prices, fast)
    ema_slow = _numba_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _numba_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

@njit(cache=True, parallel=True)
def _numba_bbands(prices: np.ndarray, length: int = 20, std: float = 2.0):
    """
    Bollinger Bands (Lower, Mid, Upper). 
    Parallelized across the sequence window for massive throughput.
    """
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    # 🚀 Rick Optimization: Parallel sliding window
    for i in prange(length - 1, n):
        mu = 0.0
        for j in range(i - length + 1, i + 1):
            mu += prices[j]
        mu /= length
        
        sigma = 0.0
        for j in range(i - length + 1, i + 1):
            sigma += (prices[j] - mu)**2
        sigma = np.sqrt(sigma / (length - 1))
        
        mid[i] = mu
        upper[i] = mu + std * sigma
        lower[i] = mu - std * sigma
        
    return lower, mid, upper

@jit(nopython=True, cache=True)
def _numba_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """Calculate Average True Range (ATR)."""
    tr = np.zeros_like(close)
    # TR[0] is High[0] - Low[0]
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        
    # Wilder's Smoothing for ATR
    atr = np.full_like(tr, np.nan)
    atr[length-1] = np.mean(tr[:length]) # Initial SMA
    
    for i in range(length, len(tr)):
        atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
        
    return atr

@jit(nopython=True, cache=True)
def _numba_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14):
    """Calculate ADX."""
    # 1. Calculate TR, +DM, -DM
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        
        # TR
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        
        # +DM
        if h_diff > l_diff and h_diff > 0:
            plus_dm[i] = h_diff
        else:
            plus_dm[i] = 0
            
        # -DM
        if l_diff > h_diff and l_diff > 0:
            minus_dm[i] = l_diff
        else:
            minus_dm[i] = 0
            
    # 2. Smooth TR, +DM, -DM using Wilder's
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    tr_smooth[length-1] = np.sum(tr[:length])
    plus_dm_smooth[length-1] = np.sum(plus_dm[:length])
    minus_dm_smooth[length-1] = np.sum(minus_dm[:length])
    
    for i in range(length, n):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/length) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/length) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/length) + minus_dm[i]
        
    # 3. Calculate DX
    dx = np.full(n, np.nan)
    for i in range(length-1, n):
        if tr_smooth[i] == 0:
            dx[i] = 0
        else:
            plus_di = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di = 100 * minus_dm_smooth[i] / tr_smooth[i]
            if plus_di + minus_di == 0:
                dx[i] = 0
            else:
                dx[i] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                
    # 4. Calculate ADX (Smoothed DX)
    adx = np.full(n, np.nan)
    # First ADX is mean of DX
    if not np.isnan(dx[2*length - 2]): # length-1 + length-1
        adx[2*length - 2] = np.mean(dx[length-1 : 2*length-1])
        
        for i in range(2*length - 1, n):
            adx[i] = (adx[i-1] * (length - 1) + dx[i]) / length
            
    return adx

# =============================================================================
# GPU-Accelerated Kernels (CUDA)
# =============================================================================

if HAS_CUDA:
    from numba import cuda
    @cuda.jit
    def _cuda_moving_average_kernel(values, window, out):
        """Massively parallel sliding window average for GPU."""
        idx = cuda.grid(1)
        if idx < values.shape[0] and idx >= window - 1:
            acc = 0.0
            for i in range(idx - window + 1, idx + 1):
                acc += values[i]
            out[idx] = acc / window

def _cuda_bbands(prices: np.ndarray, length: int = 20, std: float = 2.0):
    """GPU-accelerated Bollinger Bands."""
    if not HAS_CUDA:
        return np.full_like(prices, np.nan)
        
    n = len(prices)
    d_prices = cuda.to_device(prices.astype(np.float32))
    d_mid = cuda.device_array(n, dtype=np.float32)
    
    threadsperblock = 256
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    
    _cuda_moving_average_kernel[blockspergrid, threadsperblock](d_prices, length, d_mid)
    
    mid = d_mid.copy_to_host()
    return mid
