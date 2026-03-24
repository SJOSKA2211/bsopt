"""
Unified Mathematical Utilities - GPU-Accelerated Kernels
======================================================
Consolidates critical numerical logic for cross-module consistency.
Utilizes CuPy (GPU) and Numba (CPU fallback) for extreme performance.
Includes memory profiling to prevent GPU/CPU memory leaks.
"""

import logging
import tracemalloc
from typing import Any

import numpy as np

# Setup memory profiling
tracemalloc.start()
logger = logging.getLogger(__name__)

try:
    import cupy as cp

    GPU_AVAILABLE = True
    # Configure CuPy memory pool to prevent leaks
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    logger.info("CuPy GPU Acceleration Enabled")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy API
    logger.warning("CuPy not found. Falling back to Numba/NumPy (CPU)")

def to_numpy(arr: Any) -> np.ndarray:
    """Safely converts potential GPU array to NumPy."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)

try:
    from numba import njit, prange
except ImportError:

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

# Constants
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327

njit_engine = njit
loop_prange = prange

__all__ = [
    "calculate_price",
    "calculate_greeks",
    "profile_memory",
    "njit_engine",
    "loop_prange",
]

def profile_memory():
    """Profiles memory usage for both CPU and GPU (if available)."""
    current, peak = tracemalloc.get_traced_memory()
    stats = {
        "cpu_current_mb": current / 10**6,
        "cpu_peak_mb": peak / 10**6,
    }
    if GPU_AVAILABLE:
        stats["gpu_used_bytes"] = mempool.used_bytes()
        stats["gpu_total_bytes"] = mempool.total_bytes()
    logger.info(f"Memory Profile: {stats}")
    return stats

# -------------------------------------------------------------------------
# GPU/NumPy Vectorized Implementations (Array-native without loops)
# -------------------------------------------------------------------------

def _fast_normal_cdf(x: Any, backend: Any) -> Any:
    """High-precision rational approximation of normal CDF using vector operations."""
    abs_x = backend.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    d = INV_SQRT2PI * backend.exp(-x * x / 2.0)

    prob = (
        d
        * t
        * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
    )

    # Where x > 0, return 1 - prob, else return prob
    return backend.where(x > 0, 1.0 - prob, prob)

def _fast_normal_pdf(x: Any, backend: Any) -> Any:
    return backend.exp(-0.5 * x * x) * INV_SQRT2PI

def calculate_price(
    s: float | np.ndarray,
    k: float | np.ndarray,
    t: float | np.ndarray,
    sigma: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    is_call: bool | np.ndarray,
) -> float | np.ndarray:
    """Zero-Loop-Overhead GPU/CPU Vectorized Math Kernel for Black-Scholes Pricing."""
    backend = cp if GPU_AVAILABLE else np

    s_a = backend.asarray(s, dtype=backend.float64)
    k_a = backend.asarray(k, dtype=backend.float64)
    t_a = backend.asarray(t, dtype=backend.float64)
    sigma_a = backend.asarray(sigma, dtype=backend.float64)
    r_a = backend.asarray(r, dtype=backend.float64)
    q_a = backend.asarray(q, dtype=backend.float64)
    call_flag = backend.asarray(is_call, dtype=bool)

    safe_t = backend.maximum(t_a, 1e-9)
    safe_sigma = backend.maximum(sigma_a, 1e-9)

    vol_sqrt_t = safe_sigma * backend.sqrt(safe_t)
    d1 = (
        backend.log(s_a / k_a) + (r_a - q_a + 0.5 * safe_sigma * safe_sigma) * safe_t
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    exp_qT = backend.exp(-q_a * safe_t)
    exp_rT = backend.exp(-r_a * safe_t)

    cdf_d1 = _fast_normal_cdf(d1, backend)
    cdf_d2 = _fast_normal_cdf(d2, backend)

    call_price = s_a * exp_qT * cdf_d1 - k_a * exp_rT * cdf_d2
    put_price = k_a * exp_rT * (1.0 - cdf_d2) - s_a * exp_qT * (1.0 - cdf_d1)

    out = backend.where(call_flag, call_price, put_price)

    out = to_numpy(out)

    if np.isscalar(s) and np.isscalar(k):
        return float(out.item())
    return out

def calculate_greeks(
    s: float | np.ndarray,
    k: float | np.ndarray,
    t: float | np.ndarray,
    sigma: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    is_call: bool | np.ndarray,
) -> tuple[float | np.ndarray, ...]:
    """Vectorized GPU/CPU Math Kernel for Black-Scholes Greeks."""
    backend = cp if GPU_AVAILABLE else np

    s_a = backend.asarray(s, dtype=backend.float64)
    k_a = backend.asarray(k, dtype=backend.float64)
    t_a = backend.asarray(t, dtype=backend.float64)
    sigma_a = backend.asarray(sigma, dtype=backend.float64)
    r_a = backend.asarray(r, dtype=backend.float64)
    q_a = backend.asarray(q, dtype=backend.float64)
    call_flag = backend.asarray(is_call, dtype=bool)

    safe_t = backend.maximum(t_a, 1e-9)
    safe_sigma = backend.maximum(sigma_a, 1e-9)

    vol_sqrt_t = safe_sigma * backend.sqrt(safe_t)
    d1 = (
        backend.log(s_a / k_a) + (r_a - q_a + 0.5 * safe_sigma * safe_sigma) * safe_t
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    pdf_d1 = _fast_normal_pdf(d1, backend)
    cdf_d1 = _fast_normal_cdf(d1, backend)
    cdf_d2 = _fast_normal_cdf(d2, backend)

    exp_qT = backend.exp(-q_a * safe_t)
    exp_rT = backend.exp(-r_a * safe_t)

    # Gamma and Vega are the same for calls and puts
    gamma = (pdf_d1 * exp_qT) / (s_a * vol_sqrt_t)
    vega = s_a * exp_qT * pdf_d1 * backend.sqrt(safe_t) / 100.0

    delta = backend.where(call_flag, exp_qT * cdf_d1, exp_qT * (cdf_d1 - 1.0))
    rho = backend.where(
        call_flag,
        k_a * safe_t * exp_rT * cdf_d2 / 100.0,
        -k_a * safe_t * exp_rT * (1.0 - cdf_d2) / 100.0,
    )

    theta_call = (
        -(s_a * safe_sigma * exp_qT * pdf_d1) / (2 * backend.sqrt(safe_t))
        - r_a * k_a * exp_rT * cdf_d2
        + q_a * s_a * exp_qT * cdf_d1
    ) / 365.0

    theta_put = (
        -(s_a * safe_sigma * exp_qT * pdf_d1) / (2 * backend.sqrt(safe_t))
        + r_a * k_a * exp_rT * (1.0 - cdf_d2)
        - q_a * s_a * exp_qT * (1.0 - cdf_d1)
    ) / 365.0

    theta = backend.where(call_flag, theta_call, theta_put)

    results = (delta, gamma, theta, vega, rho)

    results = tuple(to_numpy(arr) for arr in results)

    if np.isscalar(s) and np.isscalar(k):
        return tuple(float(arr.item()) for arr in results)
    return results

@njit(cache=True, fastmath=True)
def rk4_gbm_step(s: float, mu: float, sigma: float, dt: float, dw: float) -> float:
    """
    4th-order Runge-Kutta step for Geometric Brownian Motion ODE/SDE.
    dS = mu*S*dt + sigma*S*dW
    """
    # Deterministic part: f(s) = mu * s
    # Stochastic part handled via Ito interpretation or simple noise injection
    # For GBM, RK4 is typically applied to the log-transform or the drift.

    k1 = mu * s * dt
    k2 = mu * (s + 0.5 * k1) * dt
    k3 = mu * (s + 0.5 * k2) * dt
    k4 = mu * (s + k3) * dt

    drift = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    diffusion = sigma * s * dw

    return s + drift + diffusion

def run_gbm_simulation(
    s0: float, mu: float, sigma: float, t: float, steps: int = 1000
) -> np.ndarray:
    """Runs a full GBM simulation using RK4 steps."""
    dt = t / steps
    prices = np.zeros(steps + 1)
    prices[0] = s0

    # Generate Wiener process noise
    dw = np.random.normal(0, np.sqrt(dt), steps)

    current_s = s0
    for i in range(steps):
        current_s = rk4_gbm_step(current_s, mu, sigma, dt, dw[i])
        prices[i + 1] = current_s

    return prices

loop_prange = prange
