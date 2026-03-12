"""
Unified Mathematical Utilities - Pure NumPy Vectorized
======================================================
Consolidates critical numerical logic for cross-module consistency.
Uses highly optimized pure NumPy operations without Numba loops to avoid
GIL blocking, compilation overhead, and to properly handle edge cases
(like T=0 or sigma=0) gracefully in batch operations.
"""

import os
from collections.abc import Callable
from typing import Any, Tuple, Union, TypeVar, cast, overload

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

# OPTIMIZED: Safety wrapper for environments where JIT is disabled (e.g. Test CI)
_JIT_DISABLED = os.getenv("NUMBA_DISABLE_JIT") == "1"

@overload
def njit_engine(func: F) -> F: ...

@overload
def njit_engine(*args: Any, **kwargs: Any) -> Callable[[F], F]: ...

def njit_engine(*args: Any, **kwargs: Any) -> Any:
    """
    Unified JIT decorator with proper typing and environment awareness.
    Kept for backward compatibility with other modules.
    """
    if _JIT_DISABLED:
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    try:
        from numba import njit

        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Type-safe binding of F
            f_to_jit: Any = args[0]
            return njit(cache=True, fastmath=True)(f_to_jit)

        # Return a decorator
        def decorator(f: Any) -> Any:
            kwargs.setdefault("cache", True)
            kwargs.setdefault("fastmath", True)
            return njit(*args, **kwargs)(f)

        return decorator
    except ImportError:
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

# Special case for prange which doesn't need a wrapper but needs a type-safe alias
try:
    from numba import prange as loop_prange
except ImportError:
    loop_prange = range # type: ignore

# Pre-computed constants for numerical kernels
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327
CDF_P = 0.3275911
CDF_A1 = 0.254829592
CDF_A2 = -0.284496736
CDF_A3 = 1.421413741
CDF_A4 = -1.453152027
CDF_A5 = 1.061405429

__all__ = ["njit_engine", "loop_prange", "fast_normal_cdf", "fast_normal_pdf", "calculate_d1_d2", "calculate_price", "calculate_greeks"]

def fast_normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    High-precision rational approximation of normal CDF (A&S 7.1.26).
    Fully vectorized with pure NumPy.
    """
    x = np.asarray(x, dtype=np.float64)
    # Clip extreme values to avoid overflow
    x = np.clip(x, -8.0, 8.0)
    
    abs_x = np.abs(x) * INV_SQRT2
    t = 1.0 / (1.0 + CDF_P * abs_x)
    
    # Horner's method for polynomial evaluation
    poly = t * (CDF_A1 + t * (CDF_A2 + t * (CDF_A3 + t * (CDF_A4 + t * CDF_A5))))
    y = 1.0 - poly * np.exp(-abs_x * abs_x)
    
    res = 0.5 * (1.0 + np.sign(x) * y)
    return float(res) if res.ndim == 0 else res

def fast_normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Vectorized Normal PDF."""
    x = np.asarray(x, dtype=np.float64)
    res = np.exp(-0.5 * x**2) * INV_SQRT2PI
    return float(res) if res.ndim == 0 else res

def calculate_d1_d2(
    s: Union[float, np.ndarray],
    k: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    q: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Fully vectorized d1 and d2 calculation."""
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Safe log and division using np.where
    safe_t = np.where(t > 0, t, 1e-9)
    safe_sigma = np.where(sigma > 0, sigma, 1e-9)
    sqrt_t = np.sqrt(safe_t)
    
    # Preventing ZeroDivisionError specifically on denominator
    denominator = safe_sigma * sqrt_t
    safe_denominator = np.where(denominator > 0, denominator, 1e-9)
    
    d1 = np.where(
        (t > 0) & (sigma > 0),
        (np.log(np.maximum(s / np.maximum(k, 1e-9), 1e-9)) + (r - q + 0.5 * safe_sigma**2) * safe_t) / safe_denominator,
        0.0
    )
    d2 = np.where(
        (t > 0) & (sigma > 0),
        d1 - safe_denominator,
        0.0
    )
    return (float(d1) if d1.ndim == 0 else d1, float(d2) if d2.ndim == 0 else d2)

def calculate_price(
    s: Union[float, np.ndarray],
    k: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    q: Union[float, np.ndarray],
    is_call: Union[bool, np.ndarray]
) -> Union[float, np.ndarray]:
    """Fully vectorized Black-Scholes pricing."""
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)

    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)

    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)

    call_price = s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2
    put_price = k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1)

    # Edge cases
    # t <= 0
    t_zero_call = np.maximum(s - k, 0.0)
    t_zero_put = np.maximum(k - s, 0.0)
    
    # sigma <= 0 & t > 0
    forward = s * exp_qT / exp_rT
    sigma_zero_call = np.maximum(forward - k, 0.0) * exp_rT
    sigma_zero_put = np.maximum(k - forward, 0.0) * exp_rT

    price = np.where(is_call, call_price, put_price)
    price = np.where(t <= 0, np.where(is_call, t_zero_call, t_zero_put), price)
    price = np.where((sigma <= 0) & (t > 0), np.where(is_call, sigma_zero_call, sigma_zero_put), price)

    return float(price) if price.ndim == 0 else price

def calculate_greeks(
    s: Union[float, np.ndarray],
    k: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    q: Union[float, np.ndarray],
    is_call: Union[bool, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Fully vectorized Black-Scholes Greeks calculation."""
    s = np.asarray(s, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)

    # Calculate safe d1, d2
    safe_t = np.where(t > 0, t, 1e-9)
    safe_sigma = np.where(sigma > 0, sigma, 1e-9)
    sqrt_t = np.sqrt(safe_t)
    
    d1, d2 = calculate_d1_d2(s, k, safe_t, safe_sigma, r, q)

    pdf_d1 = fast_normal_pdf(d1)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)

    exp_qT = np.exp(-q * safe_t)
    exp_rT = np.exp(-r * safe_t)

    # Standard formulas
    gamma = (pdf_d1 * exp_qT) / (s * safe_sigma * sqrt_t)
    vega = s * exp_qT * pdf_d1 * sqrt_t / 100.0

    delta = np.where(is_call, exp_qT * cdf_d1, exp_qT * (cdf_d1 - 1.0))
    rho = np.where(is_call, k * safe_t * exp_rT * cdf_d2 / 100.0, -k * safe_t * exp_rT * (1.0 - cdf_d2) / 100.0)

    theta_base = -(s * safe_sigma * exp_qT * pdf_d1) / (2 * sqrt_t)
    theta = np.where(
        is_call,
        (theta_base - r * k * exp_rT * cdf_d2 + q * s * exp_qT * cdf_d1) / 365.0,
        (theta_base + r * k * exp_rT * (1.0 - cdf_d2) - q * s * exp_qT * (1.0 - cdf_d1)) / 365.0
    )

    # Edge cases
    # t <= 0 or sigma <= 0
    edge_delta = np.where(is_call, np.where(s > k, 1.0, 0.0), np.where(s < k, -1.0, 0.0))
    edge_mask = (t <= 0) | (sigma <= 0)

    delta = np.where(edge_mask, edge_delta, delta)
    gamma = np.where(edge_mask, 0.0, gamma)
    vega = np.where(edge_mask, 0.0, vega)
    rho = np.where(edge_mask, 0.0, rho)
    theta = np.where(edge_mask, 0.0, theta)

    return (
        float(delta) if delta.ndim == 0 else delta,
        float(gamma) if gamma.ndim == 0 else gamma,
        float(theta) if theta.ndim == 0 else theta,
        float(vega) if vega.ndim == 0 else vega,
        float(rho) if rho.ndim == 0 else rho
    )

calculate_price_scalar = calculate_price
calculate_greeks_scalar = calculate_greeks
calculate_d1_d2_scalar = calculate_d1_d2
