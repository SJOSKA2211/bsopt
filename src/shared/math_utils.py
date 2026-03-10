"""
Unified Mathematical Utilities - Numba Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency with JIT acceleration.
"""

import os
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

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
    loop_prange = range

# Explicit exports for static analysis
__all__ = ["njit_engine", "loop_prange", "fast_normal_ppf", "calculate_ppf", "fast_normal_cdf", "fast_normal_pdf", "calculate_d1_d2", "calculate_price", "calculate_greeks"]

# Pre-computed constants for numerical kernels
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327
CDF_P = 0.3275911
CDF_A1 = 0.254829592
CDF_A2 = -0.284496736
CDF_A3 = 1.421413741
CDF_A4 = -1.453152027
CDF_A5 = 1.061405429


@njit_engine
def fast_normal_ppf(p: float) -> float:
    """
    Inverse CDF (PPF) approximation using Beasley-Springer-Moro.
    Optimized for JIT execution.
    """
    if p <= 0 or p >= 1:
        return 0.0

    if p < 0.5:
        # Lower tail
        return -float(_moro_inv_norm(p))
    else:
        # Upper tail
        return float(_moro_inv_norm(1.0 - p))


@njit_engine
def _moro_inv_norm(p: float) -> float:
    """Internal helper for Moro's approximation."""
    # Beasley-Springer coefficients
    a0, a1, a2, a3 = 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
    b1, b2, b3, b4 = -8.47351093090, 23.08336743743, -21.06224691826, 3.13082909833
    # Moro coefficients
    c0, c1, c2, c3, c4, c5, c6, c7, c8 = (
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    )

    y = p - 0.5
    if abs(y) < 0.42:
        # Central region
        r = y * y
        x = (
            y
            * (((a3 * r + a2) * r + a1) * r + a0)
            / ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0)
        )
        return float(x)
    else:
        # Tail region
        r = np.log(-np.log(p))
        x = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))))
        return float(x)


@njit_engine(parallel=True, nogil=True)
def _vec_ppf_impl(
    flat_p: np.ndarray[Any, np.dtype[np.float64]]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorized PPF implementation."""
    n = len(flat_p)
    flat_res = np.empty(n, dtype=np.float64)
    for i in loop_prange(n):
        flat_res[i] = fast_normal_ppf(flat_p[i])
    return flat_res


def calculate_ppf(
    p: float | np.ndarray[Any, np.dtype[np.float64]]
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    """Unified Normal PPF with Scalar Fast-Path."""
    if np.isscalar(p):
        return float(fast_normal_ppf(float(cast(float, p))))

    p_arr = np.asanyarray(p)
    if p_arr.size == 1:
        return float(fast_normal_ppf(float(p_arr.flat[0])))

    original_shape = p_arr.shape
    flat_res = _vec_ppf_impl(p_arr.ravel().astype(np.float64))
    return flat_res.reshape(original_shape)


@njit_engine
def fast_normal_cdf(x: float) -> float:
    """
    High-precision rational approximation (A&S 7.1.26).
    """
    if x > 8.0:
        return 1.0
    if x < -8.0:
        return 0.0

    abs_x = abs(x) * INV_SQRT2
    t = 1.0 / (1.0 + CDF_P * abs_x)

    # Horner's method for polynomial evaluation
    poly = t * (CDF_A1 + t * (CDF_A2 + t * (CDF_A3 + t * (CDF_A4 + t * CDF_A5))))

    y = 1.0 - poly * np.exp(-abs_x * abs_x)
    return float(0.5 * (1.0 + np.sign(x) * y))


@njit_engine
def fast_normal_pdf(x: float) -> float:
    """Numba-optimized Normal PDF."""
    return float(np.exp(-0.5 * x**2) * INV_SQRT2PI)


@njit_engine
def calculate_d1_d2(
    s: float, k: float, t: float, sigma: float, r: float, q: float
) -> tuple[float, float]:
    """Unified d1/d2 logic for scalar inputs."""
    if sigma <= 0 or t <= 0:
        return 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return float(d1), float(d2)


@njit_engine
def calculate_price_core(
    s: float, k: float, t: float, sigma: float, r: float, q: float, is_call: bool
) -> float:
    """Core Black-Scholes logic for a single element."""
    if t <= 0:
        if is_call:
            return float(max(s - k, 0.0))
        return float(max(k - s, 0.0))

    if sigma <= 0:
        df = np.exp(-r * t)
        dq = np.exp(-q * t)
        forward = s * dq / df
        if is_call:
            return float(max(forward - k, 0.0) * df)
        return float(max(k - forward, 0.0) * df)

    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)

    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)

    if is_call:
        return float(s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2)
    return float(k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1))


@njit_engine(parallel=True, nogil=True)
def _vec_price_impl(
    flat_s: np.ndarray[Any, np.dtype[np.float64]],
    flat_k: np.ndarray[Any, np.dtype[np.float64]],
    flat_t: np.ndarray[Any, np.dtype[np.float64]],
    flat_sigma: np.ndarray[Any, np.dtype[np.float64]],
    flat_r: np.ndarray[Any, np.dtype[np.float64]],
    flat_q: np.ndarray[Any, np.dtype[np.float64]],
    flat_is_call: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Vectorized price calculation."""
    n = len(flat_s)
    flat_res = np.zeros(n, dtype=np.float64)
    for i in loop_prange(n):
        flat_res[i] = calculate_price_core(
            flat_s[i],
            flat_k[i],
            flat_t[i],
            flat_sigma[i],
            flat_r[i],
            flat_q[i],
            flat_is_call[i],
        )
    return flat_res


def calculate_price(
    s: float | np.ndarray[Any, np.dtype[np.float64]],
    k: float | np.ndarray[Any, np.dtype[np.float64]],
    t: float | np.ndarray[Any, np.dtype[np.float64]],
    sigma: float | np.ndarray[Any, np.dtype[np.float64]],
    r: float | np.ndarray[Any, np.dtype[np.float64]],
    q: float | np.ndarray[Any, np.dtype[np.float64]],
    is_call: bool | np.ndarray[Any, np.dtype[np.bool_]],
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    """Unified Black-Scholes pricing with Scalar Fast-Path."""
    if (
        np.isscalar(s)
        and np.isscalar(k)
        and np.isscalar(t)
        and np.isscalar(sigma)
        and np.isscalar(r)
        and np.isscalar(q)
        and np.isscalar(is_call)
    ):
        return calculate_price_core(
            float(cast(float, s)),
            float(cast(float, k)),
            float(cast(float, t)),
            float(cast(float, sigma)),
            float(cast(float, r)),
            float(cast(float, q)),
            bool(is_call),
        )

    # Broadcast to common shape
    s_arr, k_arr, t_arr, sigma_arr, r_arr, q_arr, is_call_arr = np.broadcast_arrays(
        s, k, t, sigma, r, q, is_call
    )
    if s_arr.size == 1:
        return calculate_price_core(
            float(s_arr.flat[0]),
            float(k_arr.flat[0]),
            float(t_arr.flat[0]),
            float(sigma_arr.flat[0]),
            float(r_arr.flat[0]),
            float(q_arr.flat[0]),
            bool(is_call_arr.flat[0]),
        )

    original_shape = s_arr.shape
    flat_res = _vec_price_impl(
        s_arr.ravel().astype(np.float64),
        k_arr.ravel().astype(np.float64),
        t_arr.ravel().astype(np.float64),
        sigma_arr.ravel().astype(np.float64),
        r_arr.ravel().astype(np.float64),
        q_arr.ravel().astype(np.float64),
        is_call_arr.ravel().astype(np.bool_),
    )

    return flat_res.reshape(original_shape)


@njit_engine
def calculate_greeks_core(
    s: float, k: float, t: float, sigma: float, r: float, q: float, is_call: bool
) -> tuple[float, float, float, float, float]:
    """Core Greeks for a single element."""
    if t <= 0 or sigma <= 0:
        if is_call:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return float(delta), 0.0, 0.0, 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)

    pdf_d1 = fast_normal_pdf(d1)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)

    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)

    gamma = (pdf_d1 * exp_qT) / (s * sigma * sqrt_t)
    vega = s * exp_qT * pdf_d1 * sqrt_t / 100.0

    if is_call:
        delta = exp_qT * cdf_d1
        rho = k * t * exp_rT * cdf_d2 / 100.0
        theta_base = -(s * sigma * exp_qT * pdf_d1) / (2 * sqrt_t)
        theta = (theta_base - r * k * exp_rT * cdf_d2 + q * s * exp_qT * cdf_d1) / 365.0
    else:
        delta = exp_qT * (cdf_d1 - 1.0)
        rho = -k * t * exp_rT * (1.0 - cdf_d2) / 100.0
        theta_base = -(s * sigma * exp_qT * pdf_d1) / (2 * sqrt_t)
        theta = (
            theta_base + r * k * exp_rT * (1.0 - cdf_d2) - q * s * exp_qT * (1.0 - cdf_d1)
        ) / 365.0

    return float(delta), float(gamma), float(theta), float(vega), float(rho)


@njit_engine(parallel=True, nogil=True)
def _vec_greeks_impl(
    flat_s: np.ndarray[Any, np.dtype[np.float64]],
    flat_k: np.ndarray[Any, np.dtype[np.float64]],
    flat_t: np.ndarray[Any, np.dtype[np.float64]],
    flat_sigma: np.ndarray[Any, np.dtype[np.float64]],
    flat_r: np.ndarray[Any, np.dtype[np.float64]],
    flat_q: np.ndarray[Any, np.dtype[np.float64]],
    flat_is_call: np.ndarray[Any, np.dtype[np.bool_]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Vectorized Greeks calculation."""
    n = len(flat_s)
    f_delta = np.zeros(n, dtype=np.float64)
    f_gamma = np.zeros(n, dtype=np.float64)
    f_theta = np.zeros(n, dtype=np.float64)
    f_vega = np.zeros(n, dtype=np.float64)
    f_rho = np.zeros(n, dtype=np.float64)

    for i in loop_prange(n):
        d, g, th, v, rh = calculate_greeks_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
        f_delta[i], f_gamma[i], f_theta[i], f_vega[i], f_rho[i] = d, g, th, v, rh
    return f_delta, f_gamma, f_theta, f_vega, f_rho


def calculate_greeks(
    s: float | np.ndarray[Any, np.dtype[np.float64]],
    k: float | np.ndarray[Any, np.dtype[np.float64]],
    t: float | np.ndarray[Any, np.dtype[np.float64]],
    sigma: float | np.ndarray[Any, np.dtype[np.float64]],
    r: float | np.ndarray[Any, np.dtype[np.float64]],
    q: float | np.ndarray[Any, np.dtype[np.float64]],
    is_call: bool | np.ndarray[Any, np.dtype[np.bool_]],
) -> tuple[
    float | np.ndarray[Any, np.dtype[np.float64]],
    float | np.ndarray[Any, np.dtype[np.float64]],
    float | np.ndarray[Any, np.dtype[np.float64]],
    float | np.ndarray[Any, np.dtype[np.float64]],
    float | np.ndarray[Any, np.dtype[np.float64]],
]:
    """Unified Black-Scholes Greeks with Scalar Fast-Path."""
    if (
        np.isscalar(s)
        and np.isscalar(k)
        and np.isscalar(t)
        and np.isscalar(sigma)
        and np.isscalar(r)
        and np.isscalar(q)
        and np.isscalar(is_call)
    ):
        return calculate_greeks_core(
            float(cast(float, s)),
            float(cast(float, k)),
            float(cast(float, t)),
            float(cast(float, sigma)),
            float(cast(float, r)),
            float(cast(float, q)),
            bool(is_call),
        )

    s_arr, k_arr, t_arr, sigma_arr, r_arr, q_arr, is_call_arr = np.broadcast_arrays(
        s, k, t, sigma, r, q, is_call
    )
    if s_arr.size == 1:
        return calculate_greeks_core(
            float(s_arr.flat[0]),
            float(k_arr.flat[0]),
            float(t_arr.flat[0]),
            float(sigma_arr.flat[0]),
            float(r_arr.flat[0]),
            float(q_arr.flat[0]),
            bool(is_call_arr.flat[0]),
        )

    original_shape = s_arr.shape
    d, g, th, v, rh = _vec_greeks_impl(
        s_arr.ravel().astype(np.float64),
        k_arr.ravel().astype(np.float64),
        t_arr.ravel().astype(np.float64),
        sigma_arr.ravel().astype(np.float64),
        r_arr.ravel().astype(np.float64),
        q_arr.ravel().astype(np.float64),
        is_call_arr.ravel().astype(np.bool_),
    )

    return (
        d.reshape(original_shape),
        g.reshape(original_shape),
        th.reshape(original_shape),
        v.reshape(original_shape),
        rh.reshape(original_shape),
    )


# Aliases
calculate_price_scalar = calculate_price
calculate_greeks_scalar = calculate_greeks
calculate_d1_d2_scalar = calculate_d1_d2
