import numpy as np

try:
    from numba import config, cuda, float64, jit, njit, prange, vectorize
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
from src.shared.math_utils import calculate_price


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
    is_call_int = np.random.choice([0, 1], n_samples)
    is_call = is_call_int.astype(bool)

    # Use shared math utils - vectorized JIT calculation
    # Passing q=0.0 as implied by original kernel having no q
    q = np.zeros_like(S)
    prices = calculate_price(S, K, T, sigma, r, q, is_call)

    # Construct features
    # Note: Vectorized operations in numpy are already fast, so we keep this outside the kernel
    # unless we want to fuse everything. For now, creating the matrix is fine.
    X = np.column_stack([
        S, 
        K, 
        T, 
        is_call_int, 
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

