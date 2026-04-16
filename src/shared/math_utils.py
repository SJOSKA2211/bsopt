"""
Unified Mathematical Utilities - CPU-Optimized Kernels
==
Consolidates critical numerical logic for cross-module consistency.
Utilizes Rust (Manifold_core) and Numba (CPU) for extreme performance.
Includes memory profiling to prevent memory leaks.
"""

import logging
import tracemalloc
from typing import Any

import numpy as np

# Setup memory profiling
tracemalloc.start()
logger = logging.getLogger(__name__)

# --- Engines Detect ---
try:
    import Manifold_core as rust_core
    RUST_AVAILABLE = True
    print("[ENGINE] Rust Math Kernel (Manifold_core) ACTIVE")
    logger.info("Rust Math Kernel (Manifold_core) Enabled")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Manifold_core not found. Falling back to Python/Numba engines")


def to_numpy(arr: Any) -> np.ndarray:
    """Safely converts potential array-like to NumPy."""
    return np.asarray(arr)


import os
import sys


# --- Robust JIT Disable for Tests ---
def dummy_njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def decorator(func):
        return func
    return decorator

try:
    if os.environ.get("NUMBA_DISABLE_JIT") == "1" or os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        njit = dummy_njit
        prange = range
        logger.info("Numba JIT globally disabled for testing/env compatibility")
    else:
        from numba import njit, prange
except (ImportError, Exception):
    njit = dummy_njit
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
    """Profiles memory usage for CPU."""
    current, peak = tracemalloc.get_traced_memory()
    stats = {
        "cpu_current_mb": current / 10**6,
        "cpu_peak_mb": peak / 10**6,
    }
    logger.info(f"Memory Profile: {stats}")
    return stats


# --
# Kernels
# --

def calculate_price(
    s: float | np.ndarray,
    k: float | np.ndarray,
    t: float | np.ndarray,
    sigma: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    is_call: bool | np.ndarray,
) -> float | np.ndarray:
    """Zero-Loop-Overhead Vectorized Math Kernel for Black-Scholes Pricing."""
    
    # 1. Try Rust Core first (Highest Performance on CPU)
    if RUST_AVAILABLE:
        try:
            # Handle scalar vs array (avoid copies if already float64)
            s_a = np.asarray(s, dtype=np.float64)
            k_a = np.asarray(k, dtype=np.float64)
            t_a = np.asarray(t, dtype=np.float64)
            sigma_a = np.asarray(sigma, dtype=np.float64)
            r_a = np.asarray(r, dtype=np.float64)
            q_a = np.asarray(q, dtype=np.float64)
            call_a = np.asarray(is_call, dtype=bool)
            
            res = rust_core.batch_black_scholes(s_a, k_a, t_a, sigma_a, r_a, q_a, call_a)
            if np.isscalar(s) and np.isscalar(k):
                return float(res[0])
            return res
        except Exception as e:
            logger.error(f"Rust price kernel failed: {e}")

    # 2. Fallback to NumPy
    s_a = np.asarray(s, dtype=np.float64)
    k_a = np.asarray(k, dtype=np.float64)
    t_a = np.asarray(t, dtype=np.float64)
    sigma_a = np.asarray(sigma, dtype=np.float64)
    r_a = np.asarray(r, dtype=np.float64)
    q_a = np.asarray(q, dtype=np.float64)
    call_flag = np.asarray(is_call, dtype=bool)

    safe_t = np.maximum(t_a, 1e-9)
    safe_sigma = np.maximum(sigma_a, 1e-9)

    vol_sqrt_t = safe_sigma * np.sqrt(safe_t)
    d1 = (
        np.log(s_a / k_a) + (r_a - q_a + 0.5 * safe_sigma * safe_sigma) * safe_t
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    exp_qT = np.exp(-q_a * safe_t)
    exp_rT = np.exp(-r_a * safe_t)

    cdf_d1 = _fast_normal_cdf(d1, np)
    cdf_d2 = _fast_normal_cdf(d2, np)

    call_price = s_a * exp_qT * cdf_d1 - k_a * exp_rT * cdf_d2
    put_price = k_a * exp_rT * (1.0 - cdf_d2) - s_a * exp_qT * (1.0 - cdf_d1)

    out = np.where(call_flag, call_price, put_price)

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
    """Vectorized Math Kernel for Black-Scholes Greeks."""
    
    # 1. Try Rust Core first
    if RUST_AVAILABLE:
        try:
            s_a = np.atleast_1d(s).astype(np.float64)
            k_a = np.atleast_1d(k).astype(np.float64)
            t_a = np.atleast_1d(t).astype(np.float64)
            sigma_a = np.atleast_1d(sigma).astype(np.float64)
            r_a = np.atleast_1d(r).astype(np.float64)
            q_a = np.atleast_1d(q).astype(np.float64)
            call_a = np.atleast_1d(is_call).astype(bool)
            
            res = rust_core.batch_black_scholes_greeks(s_a, k_a, t_a, sigma_a, r_a, q_a, call_a)
            if np.isscalar(s) and np.isscalar(k):
                return tuple(float(arr[0]) for arr in res)
            return res
        except Exception as e:
            logger.error(f"Rust greeks kernel failed: {e}")

    # 2. Fallback to NumPy
    s_a = np.asarray(s, dtype=np.float64)
    k_a = np.asarray(k, dtype=np.float64)
    t_a = np.asarray(t, dtype=np.float64)
    sigma_a = np.asarray(sigma, dtype=np.float64)
    r_a = np.asarray(r, dtype=np.float64)
    q_a = np.asarray(q, dtype=np.float64)
    call_flag = np.asarray(is_call, dtype=bool)

    safe_t = np.maximum(t_a, 1e-9)
    safe_sigma = np.maximum(sigma_a, 1e-9)

    vol_sqrt_t = safe_sigma * np.sqrt(safe_t)
    d1 = (
        np.log(s_a / k_a) + (r_a - q_a + 0.5 * safe_sigma * safe_sigma) * safe_t
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    pdf_d1 = _fast_normal_pdf(d1, np)
    cdf_d1 = _fast_normal_cdf(d1, np)
    cdf_d2 = _fast_normal_cdf(d2, np)

    exp_qT = np.exp(-q_a * safe_t)
    exp_rT = np.exp(-r_a * safe_t)

    gamma = (pdf_d1 * exp_qT) / (s_a * vol_sqrt_t)
    vega = s_a * exp_qT * pdf_d1 * np.sqrt(safe_t) / 100.0

    delta = np.where(call_flag, exp_qT * cdf_d1, exp_qT * (cdf_d1 - 1.0))
    rho = np.where(
        call_flag,
        k_a * safe_t * exp_rT * cdf_d2 / 100.0,
        -k_a * safe_t * exp_rT * (1.0 - cdf_d2) / 100.0,
    )

    theta_call = (
        -(s_a * safe_sigma * exp_qT * pdf_d1) / (2 * np.sqrt(safe_t))
        - r_a * k_a * exp_rT * cdf_d2
        + q_a * s_a * exp_qT * cdf_d1
    ) / 365.0

    theta_put = (
        -(s_a * safe_sigma * exp_qT * pdf_d1) / (2 * np.sqrt(safe_t))
        + r_a * k_a * exp_rT * (1.0 - cdf_d2)
        - q_a * s_a * exp_qT * (1.0 - cdf_d1)
    ) / 365.0

    theta = np.where(call_flag, theta_call, theta_put)

    results = (delta, gamma, theta, vega, rho)

    if np.isscalar(s) and np.isscalar(k):
        return tuple(float(arr.item()) for arr in results)
    return results


def _fast_normal_cdf(x: Any, backend: Any) -> Any:
    """High-precision rational approximation of normal CDF."""
    abs_x = backend.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * abs_x)
    d = INV_SQRT2PI * backend.exp(-x * x / 2.0)
    prob = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return backend.where(x > 0, 1.0 - prob, prob)


def _fast_normal_pdf(x: Any, backend: Any) -> Any:
    return backend.exp(-0.5 * x * x) * INV_SQRT2PI



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
