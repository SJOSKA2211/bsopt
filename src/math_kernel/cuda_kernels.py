"""
GPU-Accelerated Mathematical Kernels

This module provides mathematical kernels optimized for both GPU and CPU execution:
- CUDA kernels via Numba for NVIDIA GPUs
- CuPy for GPU acceleration with CPU fallback (SIMD-optimized)
- NumPy for CPU-only execution

For CPU-only mode (AVX-512), CuPy uses optimized BLAS libraries.
"""

from __future__ import annotations

import math

import numpy as np
import structlog

try:
    from numba import cuda, float64
    from numba.core.extending import vectorize

    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None

try:
    import cupy as cp
    from cupyx.scipy.special import erf as cupy_erf

    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

from src.shared.utils.memory import profile_gpu_memory

logger = structlog.get_logger(__name__)

def _scipy_erf_approx(x: float) -> float:
    """
    High-precision rational approximation of the error function (Cody 1969).
    """
    a = [0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143, 0.0002765672, 0.0000430638]
    x_abs = abs(x)
    if x_abs > 4.0:
        return 1.0 if x > 0 else -1.0

    sum_val = 1.0
    for i, coeff in enumerate(a):
        sum_val += coeff * (x_abs ** (i + 1))

    res = 1.0 - (sum_val**-16)
    return res if x >= 0 else -res

def _get_erf_func():
    """Get erf function based on available backend."""
    if CUPY_AVAILABLE:
        return cupy_erf
    return _scipy_erf_approx

def _norm_cdf(x: float | np.ndarray | cp._core.ndarray) -> float | np.ndarray | cp._core.ndarray:
    """Vectorized normal CDF computation."""
    erf_func = _get_erf_func()
    if isinstance(x, float):
        return 0.5 * (1.0 + _scipy_erf_approx(x / math.sqrt(2)))
    return 0.5 * (1.0 + erf_func(x / math.sqrt(2)))

def _norm_pdf(x: float | np.ndarray | cp._core.ndarray) -> float | np.ndarray | cp._core.ndarray:
    """Vectorized normal PDF computation."""
    if isinstance(x, float):
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    return (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x * x)

@vectorize([float64(float64)], target="cuda")
def cnd_cuda(d: float) -> float:
    """
    Cumulative normal distribution function optimized for CUDA.
    Uses high-precision rational approximation (A&S 7.1.26) for Production accuracy.
    """
    if d < -7.0:
        return 0.0
    if d > 7.0:
        return 1.0

    # 0.5 * (1 + erf(d / sqrt(2)))
    x = d / 1.4142135623730951
    x_abs = abs(x)

    # Constants for A&S 7.1.26 (error < 1.5e-7)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)

    erf_val = y if x >= 0 else -y
    return 0.5 * (1.0 + erf_val)

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
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    d_S = cuda.to_device(S)
    d_K = cuda.to_device(K)
    d_T = cuda.to_device(T)
    d_sigma = cuda.to_device(sigma)
    d_r = cuda.to_device(r)
    d_q = cuda.to_device(q)
    d_is_call = cuda.to_device(is_call)
    d_out = cuda.device_array(n, dtype=np.float64)

    try:
        black_scholes_cuda_kernel[blocks_per_grid, threads_per_block](
            d_S, d_K, d_T, d_sigma, d_r, d_q, d_is_call, d_out
        )
        cuda.synchronize()
        out = d_out.copy_to_host()
    finally:
        del d_S, d_K, d_T, d_sigma, d_r, d_q, d_is_call, d_out

    return out

def black_scholes_cupy(
    s: np.ndarray | cp.ndarray,
    k: np.ndarray | cp.ndarray,
    t: np.ndarray | cp.ndarray,
    sigma: np.ndarray | cp.ndarray,
    r: np.ndarray | cp.ndarray,
    q: np.ndarray | cp.ndarray,
    is_call: np.ndarray | cp.ndarray,
) -> np.ndarray:
    """
    GPU-accelerated Black-Scholes option pricing via CuPy.
    Strictly typed and optimized for large-scale batches.
    """
    if CUPY_AVAILABLE:
        # Move to GPU if not already there
        s_gpu = cp.asarray(s, dtype=cp.float64)
        k_gpu = cp.asarray(k, dtype=cp.float64)
        t_gpu = cp.asarray(t, dtype=cp.float64)
        sigma_gpu = cp.asarray(sigma, dtype=cp.float64)
        r_gpu = cp.asarray(r, dtype=cp.float64)
        q_gpu = cp.asarray(q, dtype=cp.float64)
        is_call_gpu = cp.asarray(is_call, dtype=cp.bool_)

        # Guard against T <= 0
        t_pos = cp.maximum(t_gpu, 1e-10)
        sqrt_t = cp.sqrt(t_pos)

        d1 = (cp.log(s_gpu / k_gpu) + (r_gpu - q_gpu + 0.5 * sigma_gpu**2) * t_pos) / (
            sigma_gpu * sqrt_t
        )
        d2 = d1 - sigma_gpu * sqrt_t

        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)

        exp_qt = cp.exp(-q_gpu * t_pos)
        exp_rt = cp.exp(-r_gpu * t_pos)

        call_price = s_gpu * exp_qt * cdf_d1 - k_gpu * exp_rt * cdf_d2
        put_price = k_gpu * exp_rt * (1.0 - cdf_d2) - s_gpu * exp_qt * (1.0 - cdf_d1)

        # Handle T=0 explicitly
        intrinsic_call = cp.maximum(s_gpu - k_gpu, 0.0)
        intrinsic_put = cp.maximum(k_gpu - s_gpu, 0.0)

        call_price = cp.where(t_gpu <= 0, intrinsic_call, call_price)
        put_price = cp.where(t_gpu <= 0, intrinsic_put, put_price)

        result = cp.where(is_call_gpu, call_price, put_price)
        return cp.asnumpy(result)
    else:
        # CPU Fallback (Optimized NumPy)
        s = np.asanyarray(s)
        t_pos = np.maximum(t, 1e-10)
        sqrt_t = np.sqrt(t_pos)
        d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t_pos) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)
        exp_qt = np.exp(-q * t_pos)
        exp_rt = np.exp(-r * t_pos)
        call_price = s * exp_qt * cdf_d1 - k * exp_rt * cdf_d2
        put_price = k * exp_rt * (1.0 - cdf_d2) - s * exp_qt * (1.0 - cdf_d1)

        call_price = np.where(t <= 0, np.maximum(s - k, 0.0), call_price)
        put_price = np.where(t <= 0, np.maximum(k - s, 0.0), put_price)

        return np.where(is_call, call_price, put_price)

def black_scholes_greeks_cupy(
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    GPU-accelerated Black-Scholes Greeks computation via CuPy.

    Returns:
        Dictionary with delta, gamma, theta, vega, rho
    """
    if CUPY_AVAILABLE:
        s = cp.asarray(s, dtype=cp.float64)
        k = cp.asarray(k, dtype=cp.float64)
        t = cp.asarray(t, dtype=cp.float64)
        sigma = cp.asarray(sigma, dtype=cp.float64)
        r = cp.asarray(r, dtype=cp.float64)
        q = cp.asarray(q, dtype=cp.float64)
        is_call = cp.asarray(is_call, dtype=cp.bool_)

        sqrt_t = cp.sqrt(t)
        d1 = (cp.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        pdf_d1 = (1.0 / cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * d1**2)
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)

        exp_qt = cp.exp(-q * t)
        exp_rt = cp.exp(-r * t)

        call_delta = exp_qt * cdf_d1
        put_delta = exp_qt * (cdf_d1 - 1.0)
        delta = cp.where(is_call, call_delta, put_delta)

        gamma = exp_qt * pdf_d1 / (s * sigma * sqrt_t)
        vega = s * exp_qt * pdf_d1 * sqrt_t * 0.01

        theta_call = (
            (-(s * sigma * exp_qt * pdf_d1) / (2.0 * sqrt_t))
            + (q * s * exp_qt * cdf_d1)
            - (r * k * exp_rt * cdf_d2)
        )

        theta_call_per_day = theta_call / 365.0
        theta_put_per_day = (theta_call + r * k * exp_rt - q * s * exp_qt) / 365.0
        theta = cp.where(is_call, theta_call_per_day, theta_put_per_day)

        rho_call = k * t * exp_rt * cdf_d2 * 0.01
        rho_put = -k * t * exp_rt * (1 - cdf_d2) * 0.01
        rho = cp.where(is_call, rho_call, rho_put)

        return {
            "delta": cp.asnumpy(delta),
            "gamma": cp.asnumpy(gamma),
            "theta": cp.asnumpy(theta),
            "vega": cp.asnumpy(vega),
            "rho": cp.asnumpy(rho),
        }
    else:
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        pdf_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
        cdf_d1 = _norm_cdf(d1)
        cdf_d2 = _norm_cdf(d2)

        exp_qt = np.exp(-q * t)
        exp_rt = np.exp(-r * t)

        call_delta = exp_qt * cdf_d1
        put_delta = exp_qt * (cdf_d1 - 1.0)
        delta = np.where(is_call, call_delta, put_delta)

        gamma = exp_qt * pdf_d1 / (s * sigma * sqrt_t)
        vega = s * exp_qt * pdf_d1 * sqrt_t * 0.01

        theta_call = (
            (-(s * sigma * exp_qt * pdf_d1) / (2.0 * sqrt_t))
            + (q * s * exp_qt * cdf_d1)
            - (r * k * exp_rt * cdf_d2)
        )

        theta_call_per_day = theta_call / 365.0
        theta_put_per_day = (theta_call + r * k * exp_rt - q * s * exp_qt) / 365.0
        theta = np.where(is_call, theta_call_per_day, theta_put_per_day)

        rho_call = k * t * exp_rt * cdf_d2 * 0.01
        rho_put = -k * t * exp_rt * (1 - cdf_d2) * 0.01
        rho = np.where(is_call, rho_call, rho_put)

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
        }

def batch_price_cupy(
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """
    Batch pricing for large option portfolios.

    Optimized for pricing thousands of options simultaneously.
    Uses CuPy for GPU acceleration when available.
    """
    return black_scholes_cupy(s, k, t, sigma, r, q, is_call)

def batch_greeks_cupy(
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Batch Greeks computation for large option portfolios.

    Returns:
        Dictionary with delta, gamma, theta, vega, rho arrays
    """
    return black_scholes_greeks_cupy(s, k, t, sigma, r, q, is_call)

def portfolio_greeks_cupy(
    positions: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> dict[str, float]:
    """
    Calculate portfolio-level Greeks from position array.

    Args:
        positions: Position quantities (positive for long, negative for short)
        s, k, t, sigma, r, q, is_call: Standard BS parameters for each position

    Returns:
        Portfolio-level delta, gamma, theta, vega, rho
    """
    greeks = black_scholes_greeks_cupy(s, k, t, sigma, r, q, is_call)

    portfolio_greeks = {}
    for greek_name, greek_values in greeks.items():
        portfolio_greeks[f"net_{greek_name}"] = float(np.sum(positions * greek_values))

    return portfolio_greeks

if __name__ == "__main__":
    import time

    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"CuPy available: {CUPY_AVAILABLE}")

    n = 100_000
    print(f"\nBenchmarking {n:,} options...")

    s = np.random.uniform(90, 110, n)
    k = np.random.uniform(90, 110, n)
    t = np.random.uniform(0.1, 2.0, n)
    sigma = np.random.uniform(0.1, 0.5, n)
    r = np.full(n, 0.05)
    q = np.full(n, 0.02)
    is_call = np.random.choice([True, False], n)

    if CUPY_AVAILABLE:
        start = time.perf_counter()
        prices = black_scholes_cupy(s, k, t, sigma, r, q, is_call)
        elapsed = time.perf_counter() - start
        print(f"CuPy Pricing: {elapsed:.4f}s ({n / elapsed:,.0f} options/sec)")

        start = time.perf_counter()
        greeks = black_scholes_greeks_cupy(s, k, t, sigma, r, q, is_call)
        elapsed = time.perf_counter() - start
        print(f"CuPy Greeks: {elapsed:.4f}s ({n / elapsed:,.0f} options/sec)")

    positions = np.random.choice([-1, 1], n) * np.random.randint(1, 100, n)
    start = time.perf_counter()
    portfolio = portfolio_greeks_cupy(positions, s, k, t, sigma, r, q, is_call)
    elapsed = time.perf_counter() - start
    print("\nPortfolio Greeks:")
    for k, v in portfolio.items():
        print(f"  {k}: {v:.2f}")
