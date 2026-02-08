"""
Unified Mathematical Utilities - NumPy Optimized
=============================================
Consolidates critical numerical logic for cross-module consistency.
"""

import numpy as np
from scipy.special import erf


def fast_normal_cdf(x):
    """Vectorized approximation of the Normal CDF."""
    return 0.5 * (1.0 + erf(x / 1.4142135623730951))

def fast_normal_pdf(x):
    """Vectorized Normal PDF calculation."""
    return np.exp(-0.5 * x**2) / 2.5066282746310005

def calculate_d1_d2(s, k, t, sigma, r, q):
    """Unified d1/d2 logic. Handles scalars and arrays via broadcasting."""
    s, k, t, sigma, r, q = np.broadcast_arrays(s, k, t, sigma, r, q)
    sqrt_t = np.sqrt(np.maximum(t, 1e-9))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2

def calculate_price(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes pricing. Handles scalars and arrays."""
    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    
    # Handle t <= 0
    t_positive = t > 0
    prices = np.empty(s.shape, dtype=np.float64)
    
    # Positive time to expiry
    if np.any(t_positive):
        s_p = s[t_positive]
        k_p = k[t_positive]
        t_p = t[t_positive]
        sig_p = sigma[t_positive]
        r_p = r[t_positive]
        q_p = q[t_positive]
        is_call_p = is_call[t_positive]
        
        d1, d2 = calculate_d1_d2(s_p, k_p, t_p, sig_p, r_p, q_p)
        cdf_d1 = fast_normal_cdf(d1)
        cdf_d2 = fast_normal_cdf(d2)
        
        exp_qT = np.exp(-q_p * t_p)
        exp_rT = np.exp(-r_p * t_p)
        
        price_call = s_p * exp_qT * cdf_d1 - k_p * exp_rT * cdf_d2
        price_put = k_p * exp_rT * (1.0 - cdf_d2) - s_p * exp_qT * (1.0 - cdf_d1)
        
        prices[t_positive] = np.where(is_call_p, price_call, price_put)
        
    # Zero or negative time to expiry
    t_zero = ~t_positive
    if np.any(t_zero):
        prices[t_zero] = np.where(is_call[t_zero], 
                                  np.maximum(s[t_zero] - k[t_zero], 0.0),
                                  np.maximum(k[t_zero] - s[t_zero], 0.0))
        
    # Return scalar if input was scalar
    if prices.ndim == 0:
        return float(max(prices, 0.0))
    return prices

def calculate_greeks(s, k, t, sigma, r, q, is_call):
    """Unified Black-Scholes Greeks. Handles scalars and arrays."""
    s, k, t, sigma, r, q, is_call = np.broadcast_arrays(s, k, t, sigma, r, q, is_call)
    
    sqrt_t = np.sqrt(np.maximum(t, 1e-9))
    d1, d2 = calculate_d1_d2(s, k, t, sigma, r, q)
    
    INV_SQRT2PI = 1.0 / 2.5066282746310005
    
    pdf_d1 = np.exp(-0.5 * d1**2) * INV_SQRT2PI
    cdf_d1 = fast_normal_cdf(d1)
    cdf_d2 = fast_normal_cdf(d2)
    
    exp_qT = np.exp(-q * t)
    exp_rT = np.exp(-r * t)
    
    gamma = (pdf_d1 * exp_qT) / (s * sigma * sqrt_t)
    vega = s * exp_qT * pdf_d1 * sqrt_t / 100.0
    
    # Delta
    delta = np.where(is_call, exp_qT * cdf_d1, exp_qT * (cdf_d1 - 1.0))
    
    # Rho
    rho = np.where(is_call, 
                   k * t * exp_rT * cdf_d2 / 100.0,
                   -k * t * exp_rT * (1.0 - cdf_d2) / 100.0)
    
    # Theta
    theta_base = -(s * sigma * exp_qT * pdf_d1) / (2 * sqrt_t)
    theta_call = (theta_base - r * k * exp_rT * cdf_d2 + q * s * exp_qT * cdf_d1) / 365.0
    theta_put = (theta_base + r * k * exp_rT * (1.0 - cdf_d2) - q * s * exp_qT * (1.0 - cdf_d1)) / 365.0
    theta = np.where(is_call, theta_call, theta_put)
    
    # Handle scalar return
    if delta.ndim == 0:
        return float(delta), float(gamma), float(theta), float(vega), float(rho)
        
    return delta, gamma, theta, vega, rho

# Redundant scalar functions kept as aliases for backward compatibility but using unified logic
calculate_price_scalar = calculate_price
calculate_greeks_scalar = calculate_greeks
calculate_d1_d2_scalar = calculate_d1_d2
