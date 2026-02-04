import numpy as np
try:
    from numba import jit, njit, prange, config, vectorize, float64, cuda
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    class Config:
        pass
    config = Config()
    def vectorize(*args, **kwargs):
        def decorator(func):
            return np.vectorize(func)
        return decorator
    class NumbaType:
        def __call__(self, *args):
            return self
    float64 = NumbaType()
    class CudaMock:
        def jit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def grid(self, *args):
            return 0
        def device_array(self, n, dtype):
            return np.zeros(n, dtype=dtype)
    cuda = CudaMock()
import math

@jit(nopython=True, parallel=True, fastmath=True)
def _black_scholes_numba_kernel(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized kernel for Black-Scholes pricing.
    """
    n = S.shape[0]
    prices = np.zeros(n, dtype=np.float64)
    
    # Pre-calculate constants
    inv_sqrt_2 = 0.7071067811865475  # 1 / sqrt(2)
    
    for i in prange(n):
        # Handle edge cases (T ~ 0)
        if T[i] < 1e-6:
            if is_call[i] == 1:
                prices[i] = max(0.0, S[i] - K[i])
            else:
                prices[i] = max(0.0, K[i] - S[i])
            continue
            
        sqrt_T = math.sqrt(T[i])
        d1 = (math.log(S[i] / K[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * T[i]) / (sigma[i] * sqrt_T)
        d2 = d1 - sigma[i] * sqrt_T
        
        # N(x) approximation using erf
        # cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
        nd1 = 0.5 * (1.0 + math.erf(d1 * inv_sqrt_2))
        nd2 = 0.5 * (1.0 + math.erf(d2 * inv_sqrt_2))
        n_neg_d1 = 0.5 * (1.0 + math.erf(-d1 * inv_sqrt_2))
        n_neg_d2 = 0.5 * (1.0 + math.erf(-d2 * inv_sqrt_2))
        
        if is_call[i] == 1:
            prices[i] = S[i] * nd1 - K[i] * math.exp(-r[i] * T[i]) * nd2
        else:
            prices[i] = K[i] * math.exp(-r[i] * T[i]) * n_neg_d2 - S[i] * n_neg_d1
            
    return prices

def generate_synthetic_data_numba(n_samples: int = 10000, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic training data using Numba-optimized Black-Scholes engine.
    """
    np.random.seed(random_state)

    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.1, 2.0, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    is_call = np.random.choice([0, 1], n_samples)

    prices = _black_scholes_numba_kernel(S, K, T, r, sigma, is_call)

    # Construct features
    # Note: Vectorized operations in numpy are already fast, so we keep this outside the kernel
    # unless we want to fuse everything. For now, creating the matrix is fine.
    X = np.column_stack([
        S, 
        K, 
        T, 
        is_call, 
        S / K, 
        np.log(S / K), 
        np.sqrt(T), 
        T * 365, 
        sigma
    ])

    feature_names = [
        "underlying_price",
        "strike",
        "time_to_expiry",
        "is_call",
        "moneyness",
        "log_moneyness",
        "sqrt_time_to_expiry",
        "days_to_expiry",
        "implied_volatility",
    ]
    return X, prices, feature_names
