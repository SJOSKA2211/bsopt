import math

import numpy as np
import structlog
from numba import cuda, float64, vectorize

from src.utils.memory import profile_gpu_memory

logger = structlog.get_logger(__name__)


@vectorize([float64(float64)], target="cuda")
def cnd_cuda(d: float) -> float:
    """
    Cumulative normal distribution function optimized for CUDA.
    Approximation using Abramowitz and Stegun.
    """
    if d < -8.0:
        return 0.0
    if d > 8.0:
        return 1.0

    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    L = abs(d)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - RSQRT2PI * math.exp(-0.5 * L * L) * (
        A1 * K + A2 * K * K + A3 * math.pow(K, 3) + A4 * math.pow(K, 4) + A5 * math.pow(K, 5)
    )

    if d < 0.0:
        return 1.0 - w
    return w


@cuda.jit
def black_scholes_cuda_kernel(
    d_S: np.ndarray,
    d_K: np.ndarray,
    d_T: np.ndarray,
    d_sigma: np.ndarray,
    d_r: np.ndarray,
    d_q: np.ndarray,
    d_is_call: np.ndarray,
    d_out: np.ndarray,
):
    """
    CUDA Kernel for vectorized Black-Scholes pricing.
    Computes prices in parallel without Python loop overhead.
    """
    i = cuda.grid(1)
    if i < d_S.size:
        S = d_S[i]
        K = d_K[i]
        T = d_T[i]
        sigma = d_sigma[i]
        r = d_r[i]
        q = d_q[i]
        is_call = d_is_call[i]

        if T <= 0.0:
            if is_call:
                d_out[i] = max(0.0, S - K)
            else:
                d_out[i] = max(0.0, K - S)
            return

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if is_call:
            Nd1 = cnd_cuda(d1)
            Nd2 = cnd_cuda(d2)
            d_out[i] = S * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
        else:
            Nd1_neg = cnd_cuda(-d1)
            Nd2_neg = cnd_cuda(-d2)
            d_out[i] = K * math.exp(-r * T) * Nd2_neg - S * math.exp(-q * T) * Nd1_neg


@profile_gpu_memory
def price_options_gpu(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """
    Prices options utilizing the GPU. Includes memory profiling safeguards to prevent leaks.
    """
    n = S.size
    # Thread block and grid dimensions
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Copy arrays to device (memory allocation)
    d_S = cuda.to_device(S)
    d_K = cuda.to_device(K)
    d_T = cuda.to_device(T)
    d_sigma = cuda.to_device(sigma)
    d_r = cuda.to_device(r)
    d_q = cuda.to_device(q)
    d_is_call = cuda.to_device(is_call)
    d_out = cuda.device_array(n, dtype=np.float64)

    try:
        # Launch kernel
        black_scholes_cuda_kernel[blocks_per_grid, threads_per_block](
            d_S, d_K, d_T, d_sigma, d_r, d_q, d_is_call, d_out
        )

        # Explicit synchronization and copy back
        cuda.synchronize()
        out = d_out.copy_to_host()
    finally:
        # Prevent GPU memory leaks by explicitly deleting device arrays
        del d_S
        del d_K
        del d_T
        del d_sigma
        del d_r
        del d_q
        del d_is_call
        del d_out

    return out
