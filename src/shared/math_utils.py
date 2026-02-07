"""
Unified Mathematical Utilities - JIT Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency.
"""

import math

import numpy as np

try:
    from numba import float64, njit, vectorize
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def vectorize(*args, **kwargs):
        def decorator(func):
            return np.vectorize(func)
        return decorator
    class NumbaType:
        def __call__(self, *args):
            return self
    float64 = NumbaType()

@vectorize([float64(float64)], cache=True, fastmath=True)
def fast_normal_cdf(x: float) -> float:
    """Vectorized fast approximation of the Normal CDF."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

@vectorize([float64(float64)], cache=True, fastmath=True)
def fast_normal_pdf(x: float) -> float:
    """Vectorized fast Normal PDF calculation."""
    return math.exp(-0.5 * x**2) / 2.5066282746310005

def scalar_normal_cdf(x: float) -> float:
    """Scalar Normal CDF."""
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def scalar_normal_pdf(x: float) -> float:
    """Scalar Normal PDF."""
    return math.exp(-0.5 * x**2) / 2.5066282746310005

@njit(cache=True, fastmath=True)
def calculate_d1_d2(s, k, t, sigma, r, q):
    """Vectorized d1/d2 logic."""
    sqrt_t = np.sqrt(np.maximum(t, 1e-9))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2

@njit(cache=True, fastmath=True)
def calculate_price(s, k, t, sigma, r, q, is_call):
    """Vectorized Black-Scholes pricing."""
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)
    
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)
    
    call_price = s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2
    put_price = k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1)
    
    # Handle T=0 payoff
    if np.any(t <= 0):
        call_payoff = np.maximum(s - k, 0.0)
        put_payoff = np.maximum(k - s, 0.0)
        call_price = np.where(t <= 0, call_payoff, call_price)
        put_price = np.where(t <= 0, put_payoff, put_price)

    return np.where(is_call, call_price, put_price)

@njit(cache=True, fastmath=True)
def calculate_greeks(s, k, t, sigma, r, q, is_call):
    """Vectorized Black-Scholes Greeks."""
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)
    pdf_d1 = fast_normal_pdf(d1)
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)
    
    sqrt_t = np.sqrt(np.maximum(t, 1e-9))
    
    gamma = (pdf_d1 * exp_qT) / (s * sigma * sqrt_t)
    vega = s * exp_qT * pdf_d1 * sqrt_t / 100.0
    
    delta_call = exp_qT * cdf_d1
    delta_put = exp_qT * (cdf_d1 - 1.0)
    delta = np.where(is_call, delta_call, delta_put)
    
    rho_call = k * t * exp_rT * cdf_d2 / 100.0
    rho_put = -k * t * exp_rT * (1.0 - cdf_d2) / 100.0
    rho = np.where(is_call, rho_call, rho_put)
    
    theta_base = -(s * sigma * exp_qT * pdf_d1) / (2 * sqrt_t)
    theta_call = (theta_base - r * k * exp_rT * cdf_d2 + q * s * exp_qT * cdf_d1) / 365.0
    theta_put = (theta_base + r * k * exp_rT * (1.0 - cdf_d2) - q * s * exp_qT * (1.0 - cdf_d1)) / 365.0
    theta = np.where(is_call, theta_call, theta_put)
    
    return delta, gamma, theta, vega, rho

@njit(cache=True, fastmath=True)
def calculate_price_scalar(s, k, t, sigma, r, q, is_call):
    """High-performance scalar pricing."""
    if t <= 0:
        return max(s - k, 0.0) if is_call else max(k - s, 0.0)
        
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    exp_qT = math.exp(-q * t)
    exp_rT = math.exp(-r * t)
    
    # Scalar CDF approximation
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / 1.4142135623730951))
    cdf_d2 = 0.5 * (1.0 + math.erf(d2 / 1.4142135623730951))
    
    if is_call:
        return s * exp_qT * cdf_d1 - k * exp_rT * cdf_d2
    else:
        return k * exp_rT * (1.0 - cdf_d2) - s * exp_qT * (1.0 - cdf_d1)

@njit(cache=True, fastmath=True)
def calculate_greeks_scalar(s, k, t, sigma, r, q, is_call):
    """High-performance scalar greeks."""
    t_safe = max(t, 1e-9)
    sqrt_t = math.sqrt(t_safe)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t_safe) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    exp_qT = math.exp(-q * t)
    exp_rT = math.exp(-r * t)
    
    pdf_d1 = math.exp(-0.5 * d1**2) / 2.5066282746310005
    cdf_d1 = 0.5 * (1.0 + math.erf(d1 / 1.4142135623730951))
    cdf_d2 = 0.5 * (1.0 + math.erf(d2 / 1.4142135623730951))
    
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

