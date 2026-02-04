"""
Unified Mathematical Utilities - JIT Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency.
"""

import math
import numpy as np
from numba import njit, vectorize, float64

@vectorize([float64(float64)], cache=True, fastmath=True)
def fast_normal_cdf(x: float) -> float:
    """Vectorized fast approximation of the Normal CDF."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

@vectorize([float64(float64)], cache=True, fastmath=True)
def fast_normal_pdf(x: float) -> float:
    """Vectorized fast Normal PDF calculation."""
    return math.exp(-0.5 * x**2) / 2.5066282746310005

@njit(cache=True, fastmath=True)
def calculate_d1_d2(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: np.ndarray, q: np.ndarray):
    """
    Vectorized core Black-Scholes d1/d2 logic. 
    Supports NumPy arrays for high-throughput batching.
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2

@njit(cache=True, fastmath=True)
def calculate_d1_d2_scalar(S: float, K: float, T: float, sigma: float, r: float, q: float):
    """Scalar version for minimal overhead on single calls."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2

