"""
Unified Mathematical Utilities - Numba Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency with JIT acceleration.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def fast_normal_cdf(x):
    """
    Numba-compatible Normal CDF approximation.
    Uses a high-precision rational approximation (A&S 7.1.26).
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    abs_x = abs(x) / 1.4142135623730951
    t = 1.0 / (1.0 + p * abs_x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-abs_x * abs_x)
    
    res = 0.5 * (1.0 + np.sign(x) * y)
    return res

@njit(cache=True, fastmath=True)
def fast_normal_pdf(x):
    """Numba-optimized Normal PDF."""
    return np.exp(-0.5 * x**2) / 2.5066282746310005

@njit(cache=True, fastmath=True)
def calculate_d1_d2(s, k, t, sigma, r, q):
    """Unified d1/d2 logic for scalar inputs."""
    sqrt_t = np.sqrt(max(t, 1e-9))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2

@njit(cache=True, fastmath=True)
def calculate_price_core(s, k, t, sigma, r, q, is_call):
    """Core Black-Scholes logic for a single element."""
    if t <= 0:
        if is_call:
            return max(s - k, 0.0)
        else:
            return max(k - s, 0.0)
            
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)
    
    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)
    
    if is_call:
        return s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2
    else:
        return k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1)


@njit(cache=True, fastmath=True)
def _vec_price_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call):
    """Vectorized price calculation (JIT-compiled loop)."""
    n = len(flat_s)
    flat_res = np.empty(n, dtype=np.float64)
    for i in range(n):
        flat_res[i] = calculate_price_core(
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], 
            flat_r[i], flat_q[i], flat_is_call[i]
        )
    return flat_res


def calculate_price(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes pricing. Vectorized via NumPy/Numba."""
    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    
    original_shape = s.shape
    flat_s = np.ascontiguousarray(s.ravel()).astype(np.float64)
    flat_k = np.ascontiguousarray(k.ravel()).astype(np.float64)
    flat_t = np.ascontiguousarray(t.ravel()).astype(np.float64)
    flat_sigma = np.ascontiguousarray(sigma.ravel()).astype(np.float64)
    flat_r = np.ascontiguousarray(r.ravel()).astype(np.float64)
    flat_q = np.ascontiguousarray(q.ravel()).astype(np.float64)
    flat_is_call = np.ascontiguousarray(is_call.ravel()).astype(np.bool_)

    flat_res = _vec_price_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call)
    res = flat_res.reshape(original_shape)
    
    if res.ndim == 0 or res.size == 1:
        return float(max(res.flat[0], 0.0))
    return res


@njit(cache=True, fastmath=True)
def calculate_greeks_core(s, k, t, sigma, r, q, is_call):
    """Core Greeks for a single element."""
    sqrt_t = np.sqrt(max(t, 1e-9))
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
        theta = (theta_base + r * k * exp_rT * (1.0 - cdf_d2) - q * s * exp_qT * (1.0 - cdf_d1)) / 365.0
        
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
            flat_s[i], flat_k[i], flat_t[i], flat_sigma[i], 
            flat_r[i], flat_q[i], flat_is_call[i]
        )
        f_delta[i] = d
        f_gamma[i] = g
        f_theta[i] = th
        f_vega[i] = v
        f_rho[i] = rh
        
    return f_delta, f_gamma, f_theta, f_vega, f_rho


def calculate_greeks(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes Greeks. Vectorized via Numba."""
    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    
    original_shape = s.shape
    flat_s = np.ascontiguousarray(s.ravel()).astype(np.float64)
    flat_k = np.ascontiguousarray(k.ravel()).astype(np.float64)
    flat_t = np.ascontiguousarray(t.ravel()).astype(np.float64)
    flat_sigma = np.ascontiguousarray(sigma.ravel()).astype(np.float64)
    flat_r = np.ascontiguousarray(r.ravel()).astype(np.float64)
    flat_q = np.ascontiguousarray(q.ravel()).astype(np.float64)
    flat_is_call = np.ascontiguousarray(is_call.ravel()).astype(np.bool_)

    d, g, th, v, rh = _vec_greeks_impl(flat_s, flat_k, flat_t, flat_sigma, flat_r, flat_q, flat_is_call)
    
    delta = d.reshape(original_shape)
    gamma = g.reshape(original_shape)
    theta = th.reshape(original_shape)
    vega = v.reshape(original_shape)
    rho = rh.reshape(original_shape)
    
    if delta.ndim == 0 or delta.size == 1:
        return float(delta.flat[0]), float(gamma.flat[0]), float(theta.flat[0]), float(vega.flat[0]), float(rho.flat[0])
    return delta, gamma, theta, vega, rho


# Aliases for backward compatibility
calculate_price_scalar = calculate_price
calculate_greeks_scalar = calculate_greeks
calculate_d1_d2_scalar = calculate_d1_d2
