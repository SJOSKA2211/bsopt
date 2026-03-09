"""
Unified Mathematical Utilities - Numba Optimized 🚀
=============================================
Consolidates critical numerical logic for cross-module consistency with JIT acceleration.
"""

import os

import numpy as np

# OPTIMIZED: Safety wrapper for environments where JIT is disabled (e.g. Test CI)
if os.getenv("NUMBA_DISABLE_JIT") == "1":

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    _prange = range
    njit = _njit  # For external compatibility
    prange = _prange
else:
    try:
        from numba import njit, prange

        # Enable AOT caching and fastmath by default to eliminate cold-starts and push limits
        def _njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return njit(cache=True, fastmath=True)(args[0])
            kwargs.setdefault("cache", True)
            kwargs.setdefault("fastmath", True)
            return njit(*args, **kwargs)

        _prange = prange
    except ImportError:

        def _njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda f: f

        _prange = range
        njit = _njit
        prange = _prange

# Pre-computed constants for numerical kernels
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327
CDF_P = 0.3275911
CDF_A1 = 0.254829592
CDF_A2 = -0.284496736
CDF_A3 = 1.421413741
CDF_A4 = -1.453152027
CDF_A5 = 1.061405429


@_njit
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
    return 0.5 * (1.0 + np.sign(x) * y)


@_njit
def fast_normal_pdf(x: float) -> float:
    """Numba-optimized Normal PDF."""
    return np.exp(-0.5 * x**2) * INV_SQRT2PI


@_njit
def calculate_d1_d2(
    s: float, k: float, t: float, sigma: float, r: float, q: float
) -> tuple[float, float]:
    """Unified d1/d2 logic for scalar inputs."""
    if sigma <= 0 or t <= 0:
        return 0.0, 0.0

    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


@_njit
def calculate_price_core(
    s: float, k: float, t: float, sigma: float, r: float, q: float, is_call: bool
) -> float:
    """Core Black-Scholes logic for a single element."""
    if t <= 0:
        if is_call:
            return max(s - k, 0.0)
        return max(k - s, 0.0)

    if sigma <= 0:
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


@_njit
def _vec_price_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    """Vectorized price calculation."""
    n = len(flat_s)
    flat_res = np.zeros(n, dtype=np.float64)
    for i in _prange(n):
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
            float(s), float(k), float(t), float(sigma), float(r), float(q), bool(is_call)
        )

    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    if s.size == 1:
        return calculate_price_core(
            float(s.flat[0]),
            float(k.flat[0]),
            float(t.flat[0]),
            float(sigma.flat[0]),
            float(r.flat[0]),
            float(q.flat[0]),
            bool(is_call.flat[0]),
        )

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
    return flat_res.reshape(original_shape)


@_njit
def calculate_greeks_core(
    s: float, k: float, t: float, sigma: float, r: float, q: float, is_call: bool
) -> tuple[float, float, float, float, float]:
    """Core Greeks for a single element."""
    if t <= 0 or sigma <= 0:
        if is_call:
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return delta, 0.0, 0.0, 0.0, 0.0

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

    return delta, gamma, theta, vega, rho


@_njit
def _vec_greeks_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    """Vectorized Greeks calculation."""
    n = len(flat_s)
    f_delta = np.zeros(n, dtype=np.float64)
    f_gamma = np.zeros(n, dtype=np.float64)
    f_theta = np.zeros(n, dtype=np.float64)
    f_vega = np.zeros(n, dtype=np.float64)
    f_rho = np.zeros(n, dtype=np.float64)

    for i in _prange(n):
        d, g, th, v, rh = calculate_greeks_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], flat_r[i], flat_q[i], flat_is_call[i]
        )
        f_delta[i], f_gamma[i], f_theta[i], f_vega[i], f_rho[i] = d, g, th, v, rh
    return f_delta, f_gamma, f_theta, f_vega, f_rho


def calculate_greeks(s, k, t, sigma, r, q, is_call):
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
            float(s), float(k), float(t), float(sigma), float(r), float(q), bool(is_call)
        )

    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    if s.size == 1:
        return calculate_greeks_core(
            float(s.flat[0]),
            float(k.flat[0]),
            float(t.flat[0]),
            float(sigma.flat[0]),
            float(r.flat[0]),
            float(q.flat[0]),
            bool(is_call.flat[0]),
        )

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
