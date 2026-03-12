"""
Unified Mathematical Utilities - Optimized Numba Vectorized Kernels
======================================================
Consolidates critical numerical logic for cross-module consistency.
Uses highly optimized Numba C-based JIT kernels to process equations 
across the entire dataset without Python loop overhead.
"""

import os
from typing import Any, Tuple, Union
import numpy as np

try:
    from numba import njit, prange
except ImportError:
    # Fallback if Numba isn't installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Pre-computed constants for numerical kernels
INV_SQRT2 = 0.7071067811865476
INV_SQRT2PI = 0.3989422804014327
CDF_P = 0.3275911
CDF_A1 = 0.254829592
CDF_A2 = -0.284496736
CDF_A3 = 1.421413741
CDF_A4 = -1.453152027
CDF_A5 = 1.061405429

__all__ = ["fast_normal_cdf", "fast_normal_pdf", "calculate_d1_d2", "calculate_price", "calculate_greeks"]

@njit(fastmath=True, cache=True)
def fast_normal_cdf_scalar(x: float) -> float:
    """High-precision rational approximation of normal CDF."""
    if x < -8.0: return 0.0
    if x > 8.0: return 1.0
    
    abs_x = abs(x) * INV_SQRT2
    t = 1.0 / (1.0 + CDF_P * abs_x)
    
    poly = t * (CDF_A1 + t * (CDF_A2 + t * (CDF_A3 + t * (CDF_A4 + t * CDF_A5))))
    y = 1.0 - poly * np.exp(-abs_x * abs_x)
    
    return 0.5 * (1.0 + np.sign(x) * y)

@njit(fastmath=True, cache=True)
def fast_normal_pdf_scalar(x: float) -> float:
    return np.exp(-0.5 * x**2) * INV_SQRT2PI

@njit(fastmath=True, cache=True)
def calculate_price_kernel(
    s: np.ndarray, k: np.ndarray, t: np.ndarray, 
    sigma: np.ndarray, r: np.ndarray, q: np.ndarray, 
    is_call: np.ndarray, out: np.ndarray
):
    """Vectorized C-based Math Kernel for Black-Scholes Pricing using Numba."""
    n = s.shape[0]
    for i in prange(n):
        si = s[i]
        ki = k[i]
        ti = t[i]
        sig_i = sigma[i]
        ri = r[i]
        qi = q[i]
        call_flag = is_call[i]

        if ti <= 0.0:
            if call_flag:
                out[i] = max(si - ki, 0.0)
            else:
                out[i] = max(ki - si, 0.0)
            continue
            
        if sig_i <= 0.0:
            forward = si * np.exp(-qi * ti) / np.exp(-ri * ti)
            if call_flag:
                out[i] = max(forward - ki, 0.0) * np.exp(-ri * ti)
            else:
                out[i] = max(ki - forward, 0.0) * np.exp(-ri * ti)
            continue

        denominator = sig_i * np.sqrt(ti)
        d1 = (np.log(max(si / max(ki, 1e-9), 1e-9)) + (ri - qi + 0.5 * sig_i**2) * ti) / denominator
        d2 = d1 - denominator

        cdf_d1 = fast_normal_cdf_scalar(d1)
        cdf_d2 = fast_normal_cdf_scalar(d2)

        exp_qT = np.exp(-qi * ti)
        exp_rT = np.exp(-ri * ti)

        if call_flag:
            out[i] = si * exp_qT * cdf_d1 - ki * exp_rT * cdf_d2
        else:
            out[i] = ki * exp_rT * (1.0 - cdf_d2) - si * exp_qT * (1.0 - cdf_d1)

@njit(fastmath=True, cache=True)
def calculate_greeks_kernel(
    s: np.ndarray, k: np.ndarray, t: np.ndarray, 
    sigma: np.ndarray, r: np.ndarray, q: np.ndarray, 
    is_call: np.ndarray, 
    out_delta: np.ndarray, out_gamma: np.ndarray, 
    out_theta: np.ndarray, out_vega: np.ndarray, out_rho: np.ndarray
):
    """Vectorized C-based Math Kernel for Black-Scholes Greeks using Numba."""
    n = s.shape[0]
    for i in prange(n):
        si = s[i]
        ki = k[i]
        ti = max(t[i], 1e-9)
        sig_i = max(sigma[i], 1e-9)
        ri = r[i]
        qi = q[i]
        call_flag = is_call[i]

        is_edge = (t[i] <= 0) or (sigma[i] <= 0)

        if is_edge:
            out_gamma[i] = 0.0
            out_vega[i] = 0.0
            out_rho[i] = 0.0
            out_theta[i] = 0.0
            if call_flag:
                out_delta[i] = 1.0 if si > ki else 0.0
            else:
                out_delta[i] = -1.0 if si < ki else 0.0
            continue

        denominator = sig_i * np.sqrt(ti)
        d1 = (np.log(max(si / max(ki, 1e-9), 1e-9)) + (ri - qi + 0.5 * sig_i**2) * ti) / denominator
        d2 = d1 - denominator

        pdf_d1 = fast_normal_pdf_scalar(d1)
        cdf_d1 = fast_normal_cdf_scalar(d1)
        cdf_d2 = fast_normal_cdf_scalar(d2)

        exp_qT = np.exp(-qi * ti)
        exp_rT = np.exp(-ri * ti)
        sqrt_t = np.sqrt(ti)

        out_gamma[i] = (pdf_d1 * exp_qT) / (si * sig_i * sqrt_t)
        out_vega[i] = si * exp_qT * pdf_d1 * sqrt_t / 100.0

        theta_base = -(si * sig_i * exp_qT * pdf_d1) / (2 * sqrt_t)

        if call_flag:
            out_delta[i] = exp_qT * cdf_d1
            out_rho[i] = ki * ti * exp_rT * cdf_d2 / 100.0
            out_theta[i] = (theta_base - ri * ki * exp_rT * cdf_d2 + qi * si * exp_qT * cdf_d1) / 365.0
        else:
            out_delta[i] = exp_qT * (cdf_d1 - 1.0)
            out_rho[i] = -ki * ti * exp_rT * (1.0 - cdf_d2) / 100.0
            out_theta[i] = (theta_base + ri * ki * exp_rT * (1.0 - cdf_d2) - qi * si * exp_qT * (1.0 - cdf_d1)) / 365.0

# Thin Wrappers to maintain exact signature compatibility with bsopt-api
def fast_normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(x, (float, int)):
        return fast_normal_cdf_scalar(float(x))
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    # Numba vectorization of scalar functions
    for i in range(x_arr.size):
        out.flat[i] = fast_normal_cdf_scalar(x_arr.flat[i])
    return out

def fast_normal_pdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(x, (float, int)):
        return fast_normal_pdf_scalar(float(x))
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    for i in range(x_arr.size):
        out.flat[i] = fast_normal_pdf_scalar(x_arr.flat[i])
    return out

def calculate_d1_d2(
    s: Union[float, np.ndarray], k: Union[float, np.ndarray], t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray], r: Union[float, np.ndarray], q: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    # Simple python wrapper since the actual D1/D2 calculation happens natively in the big kernels anyway
    s_a = np.atleast_1d(s).astype(np.float64)
    k_a = np.atleast_1d(k).astype(np.float64)
    t_a = np.atleast_1d(t).astype(np.float64)
    sigma_a = np.atleast_1d(sigma).astype(np.float64)
    r_a = np.atleast_1d(r).astype(np.float64)
    q_a = np.atleast_1d(q).astype(np.float64)

    target_shape = np.broadcast(s_a, k_a, t_a, sigma_a, r_a, q_a).shape
    safe_t = np.where(t_a > 0, t_a, 1e-9)
    safe_sigma = np.where(sigma_a > 0, sigma_a, 1e-9)
    denominator = safe_sigma * np.sqrt(safe_t)
    
    d1 = np.where((t_a > 0) & (sigma_a > 0), (np.log(np.maximum(s_a / np.maximum(k_a, 1e-9), 1e-9)) + (r_a - q_a + 0.5 * safe_sigma**2) * safe_t) / denominator, 0.0)
    d2 = np.where((t_a > 0) & (sigma_a > 0), d1 - denominator, 0.0)
    
    if np.isscalar(s) and np.isscalar(k) and np.isscalar(t) and np.isscalar(sigma) and np.isscalar(r) and np.isscalar(q):
        return float(d1[0]), float(d2[0])
    return d1, d2

def calculate_price(
    s: Union[float, np.ndarray], k: Union[float, np.ndarray], t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray], r: Union[float, np.ndarray], q: Union[float, np.ndarray],
    is_call: Union[bool, np.ndarray]
) -> Union[float, np.ndarray]:
    s_a = np.atleast_1d(s).astype(np.float64)
    k_a = np.atleast_1d(k).astype(np.float64)
    t_a = np.atleast_1d(t).astype(np.float64)
    sigma_a = np.atleast_1d(sigma).astype(np.float64)
    r_a = np.atleast_1d(r).astype(np.float64)
    q_a = np.atleast_1d(q).astype(np.float64)
    is_call_a = np.atleast_1d(is_call).astype(bool)

    target_shape = np.broadcast(s_a, k_a, t_a, sigma_a, r_a, q_a, is_call_a).shape
    
    def _bcast(arr):
        return np.broadcast_to(arr, target_shape).flatten()
        
    s_f, k_f, t_f, sig_f, r_f, q_f, call_f = map(_bcast, (s_a, k_a, t_a, sigma_a, r_a, q_a, is_call_a))
    out = np.empty_like(s_f)
    
    calculate_price_kernel(s_f, k_f, t_f, sig_f, r_f, q_f, call_f, out)
    
    out = out.reshape(target_shape)
    if isinstance(s, (float, int)) and isinstance(k, (float, int)):
        return float(out.item())
    return out

def calculate_greeks(
    s: Union[float, np.ndarray], k: Union[float, np.ndarray], t: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray], r: Union[float, np.ndarray], q: Union[float, np.ndarray],
    is_call: Union[bool, np.ndarray]
) -> Tuple[Union[float, np.ndarray], ...]:
    s_a = np.atleast_1d(s).astype(np.float64)
    k_a = np.atleast_1d(k).astype(np.float64)
    t_a = np.atleast_1d(t).astype(np.float64)
    sigma_a = np.atleast_1d(sigma).astype(np.float64)
    r_a = np.atleast_1d(r).astype(np.float64)
    q_a = np.atleast_1d(q).astype(np.float64)
    is_call_a = np.atleast_1d(is_call).astype(bool)

    target_shape = np.broadcast(s_a, k_a, t_a, sigma_a, r_a, q_a, is_call_a).shape
    
    def _bcast(arr):
        return np.broadcast_to(arr, target_shape).flatten()
        
    s_f, k_f, t_f, sig_f, r_f, q_f, call_f = map(_bcast, (s_a, k_a, t_a, sigma_a, r_a, q_a, is_call_a))
    
    d, g, th, v, rh = np.empty_like(s_f), np.empty_like(s_f), np.empty_like(s_f), np.empty_like(s_f), np.empty_like(s_f)
    
    calculate_greeks_kernel(s_f, k_f, t_f, sig_f, r_f, q_f, call_f, d, g, th, v, rh)
    
    d, g, th, v, rh = [arr.reshape(target_shape) for arr in (d, g, th, v, rh)]
    
    if isinstance(s, (float, int)) and isinstance(k, (float, int)):
        return float(d.item()), float(g.item()), float(th.item()), float(v.item()), float(rh.item())
    return d, g, th, v, rh

def njit_engine(*args, **kwargs):
    def decorator(f):
        return f
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

loop_prange = prange
