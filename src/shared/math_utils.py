"""
Unified Mathematical Utilities - Numba Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency with JIT acceleration.
"""

import numpy as np
from numba import njit

# OPTIMIZED: Pre-computed constants for numerical kernels
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327
CDF_P = 0.3275911
CDF_A = np.array([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])


@njit(cache=True, fastmath=True)
def fast_normal_cdf(x):
    """
    High-precision rational approximation (A&S 7.1.26).
    OPTIMIZED: Horner's method and pre-computed constants.
    """
    if x > 8.0:
        return 1.0
    if x < -8.0:
        return 0.0

    abs_x = abs(x) * INV_SQRT2
    t = 1.0 / (1.0 + CDF_P * abs_x)

    # Horner's method for polynomial evaluation: a1*t + a2*t^2 + ...
    # poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    poly = t * (CDF_A[0] + t * (CDF_A[1] + t * (CDF_A[2] + t * (CDF_A[3] + t * CDF_A[4]))))

    y = 1.0 - poly * np.exp(-abs_x * abs_x)
    return 0.5 * (1.0 + np.sign(x) * y)


@njit(cache=True, fastmath=True)
def fast_normal_pdf(x):
    """Numba-optimized Normal PDF."""
    return np.exp(-0.5 * x**2) * INV_SQRT2PI


@njit(cache=True, fastmath=True)
def calculate_d1_d2(s, k, t, sigma, r, q):
    """Unified d1/d2 logic for scalar inputs."""
    # Handle zero sigma or maturity gracefully to avoid NaN
    if sigma <= 0 or t <= 0:
        return 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


@njit(cache=True, fastmath=True)
def calculate_price_core(s, k, t, sigma, r, q, is_call):
    """Core Black-Scholes logic for a single element."""
    if t <= 0:
        if is_call:
            return max(s - k, 0.0)
        return max(k - s, 0.0)

    if sigma <= 0:
        # Zero volatility case: risk-free growth
        df = np.exp(-r * t)
        dq = np.exp(-q * t)
        forward = s * dq / df
        if is_call:
            return max(forward - k, 0.0) * df
        return max(k - forward, 0.0) * df

    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)

    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)

    if is_call:
        return s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2
    return k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1)


@njit(cache=True, fastmath=True)
def _vec_price_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    """Vectorized price calculation (JIT-compiled loop)."""
    n = len(flat_s)
    flat_res = np.empty(n, dtype=np.float64)
    for i in range(n):
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


def calculate_price(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes pricing. Vectorized via NumPy/Numba with Scalar Fast-Path."""
    # Handle scalar types explicitly
    if (
        np.isscalar(s)
        and np.isscalar(k)
        and np.isscalar(t)
        and np.isscalar(sigma)
        and np.isscalar(r)
        and np.isscalar(q)
        and np.isscalar(is_call)
    ):
        res = calculate_price_core(
            float(s),
            float(k),
            float(t),
            float(sigma),
            float(r),
            float(q),
            bool(is_call),
        )
        return float(max(res, 0.0))

    # Vectorized path
    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)

    if s.size == 0:
        return np.array([], dtype=np.float64)

    original_shape = s.shape
    flat_res = _vec_price_impl(
        s.ravel().astype(np.float64),
        k.ravel().astype(np.float64),
        t.ravel().astype(np.float64),
        sigma.ravel().astype(np.float64),
        r.ravel().astype(np.float64),
        q.ravel().astype(np.float64),
        is_call.ravel().astype(np.bool_),
    )

    res = flat_res.reshape(original_shape)
    if res.size == 1:
        return float(max(res.item(), 0.0))
    return res


@njit(cache=True, fastmath=True)
def calculate_greeks_core(s, k, t, sigma, r, q, is_call):
    """Core Greeks for a single element."""
    if t <= 0 or sigma <= 0:
        # Simplified boundary Greeks
        if is_call:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return delta, 0.0, 0.0, 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)

    INV_SQRT2PI = 1.0 / 2.5066282746310005

    pdf_d1 = np.exp(-0.5 * d1**2) * INV_SQRT2PI
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

    return delta, gamma, theta, vega, rho


@njit(cache=True, fastmath=True)
def _vec_greeks_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    """Vectorized Greeks calculation (JIT-compiled loop)."""
    n = len(flat_s)
    f_delta = np.empty(n, dtype=np.float64)
    f_gamma = np.empty(n, dtype=np.float64)
    f_theta = np.empty(n, dtype=np.float64)
    f_vega = np.empty(n, dtype=np.float64)
    f_rho = np.empty(n, dtype=np.float64)

    for i in range(n):
        d, g, th, v, rh = calculate_greeks_core(
            flat_s[i],
            flat_k[i],
            flat_t[i],
            flat_sigma[i],
            flat_r[i],
            flat_q[i],
            flat_is_call[i],
        )
        f_delta[i] = d
        f_gamma[i] = g
        f_theta[i] = th
        f_vega[i] = v
        f_rho[i] = rh

    return f_delta, f_gamma, f_theta, f_vega, f_rho


def calculate_greeks(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes Greeks. Vectorized via Numba with Scalar Fast-Path."""
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
            float(s),
            float(k),
            float(t),
            float(sigma),
            float(r),
            float(q),
            bool(is_call),
        )

    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)

    if s.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    original_shape = s.shape
    d, g, th, v, rh = _vec_greeks_impl(
        s.ravel().astype(np.float64),
        k.ravel().astype(np.float64),
        t.ravel().astype(np.float64),
        sigma.ravel().astype(np.float64),
        r.ravel().astype(np.float64),
        q.ravel().astype(np.float64),
        is_call.ravel().astype(np.bool_),
    )

    if s.size == 1:
        return (
            float(d.item()),
            float(g.item()),
            float(th.item()),
            float(v.item()),
            float(rh.item()),
        )

    return (
        d.reshape(original_shape),
        g.reshape(original_shape),
        th.reshape(original_shape),
        v.reshape(original_shape),
        rh.reshape(original_shape),
    )


# Aliases for backward compatibility
calculate_price_scalar = calculate_price
calculate_greeks_scalar = calculate_greeks
calculate_d1_d2_scalar = calculate_d1_d2
