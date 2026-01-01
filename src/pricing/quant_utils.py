"""
Highly Optimized Quantitative Implementation Utilities

This module provides low-level performance-critical utilities for numerical
finance calculations, including fast initial guesses and JIT-compiled math.
"""

import math
from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(cache=True)
def fast_normal_cdf(x: float) -> float:
    """
    Fast approximation of the Normal CDF using math.erf.
    """
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))


@njit(cache=True)
def fast_normal_pdf(x: float) -> float:
    """
    Fast Normal PDF calculation.
    """
    return math.exp(-0.5 * x**2) / 2.5066282746310005  # sqrt(2*pi)


@njit(parallel=True, cache=True)
def corrado_miller_initial_guess(
    market_price: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend: np.ndarray,
    option_type: np.ndarray,
) -> np.ndarray:
    """
    Fast initial guess for Implied Volatility using Corrado-Miller approximation.
    """
    n = len(market_price)
    sigma = np.empty(n, dtype=np.float64)

    for i in prange(n):
        S = spot[i]
        K = strike[i]
        T = maturity[i]
        r = rate[i]
        q = dividend[i]
        P = market_price[i]

        X = K * math.exp(-r * T)
        val = 2.5066282746310005 / (math.sqrt(T) * (S + X))

        exp_qt = math.exp(-q * T)
        if option_type[i] == 0:
            intrinsic = max(S * exp_qt - X, 0.0)
        else:
            intrinsic = max(X - S * exp_qt, 0.0)

        term = P - intrinsic / 2.0
        sigma[i] = val * (term + math.sqrt(max(term**2 - intrinsic**2 / 3.141592653589793, 0.0)))

    return np.clip(sigma, 0.001, 5.0)


@njit(cache=True)
def calculate_d1_d2_jit(
    S: float, K: float, T: float, sigma: float, r: float, q: float
) -> Tuple[float, float]:
    """JIT compiled Black-Scholes d1 and d2."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


@njit(parallel=True, cache=True)
def batch_bs_price_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """JIT compiled batch pricing for options."""
    n = len(S)
    prices = np.empty(n, dtype=np.float64)

    for i in prange(n):
        if T[i] < 1e-7:
            if is_call[i]:
                prices[i] = max(S[i] - K[i], 0.0)
            else:
                prices[i] = max(K[i] - S[i], 0.0)
            continue

        d1, d2 = calculate_d1_d2_jit(S[i], K[i], T[i], sigma[i], r[i], q[i])

        nd1 = 0.5 * (1.0 + math.erf(d1 / 1.4142135623730951))
        nd2 = 0.5 * (1.0 + math.erf(d2 / 1.4142135623730951))

        if is_call[i]:
            prices[i] = S[i] * math.exp(-q[i] * T[i]) * nd1 - K[i] * math.exp(-r[i] * T[i]) * nd2
        else:
            prices[i] = K[i] * math.exp(-r[i] * T[i]) * (1.0 - nd2) - S[i] * math.exp(
                -q[i] * T[i]
            ) * (1.0 - nd1)

    return np.maximum(prices, 0.0)


@njit(parallel=True, cache=True)
def batch_greeks_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """JIT compiled batch greeks calculation."""
    n = len(S)
    delta = np.empty(n, dtype=np.float64)
    gamma = np.empty(n, dtype=np.float64)
    vega = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)

    inv_sqrt_2pi = 1.0 / 2.5066282746310005

    for i in prange(n):
        Ti = max(T[i], 1e-7)
        sqrt_T = math.sqrt(Ti)
        d1, d2 = calculate_d1_d2_jit(S[i], K[i], Ti, sigma[i], r[i], q[i])

        # PDF and CDF
        pdf_d1 = math.exp(-0.5 * d1**2) * inv_sqrt_2pi
        cdf_d1 = 0.5 * (1.0 + math.erf(d1 / 1.4142135623730951))
        cdf_d2 = 0.5 * (1.0 + math.erf(d2 / 1.4142135623730951))

        exp_qt = math.exp(-q[i] * Ti)
        exp_rt = math.exp(-r[i] * Ti)

        # Delta
        if is_call[i]:
            delta[i] = exp_qt * cdf_d1
        else:
            delta[i] = exp_qt * (cdf_d1 - 1.0)

        # Gamma
        gamma[i] = (exp_qt * pdf_d1) / (S[i] * sigma[i] * sqrt_T)

        # Vega
        vega[i] = (S[i] * exp_qt * pdf_d1 * sqrt_T) * 0.01

        # Theta
        common_theta = -(S[i] * pdf_d1 * sigma[i] * exp_qt) / (2 * sqrt_T)
        if is_call[i]:
            theta[i] = (
                common_theta - r[i] * K[i] * exp_rt * cdf_d2 + q[i] * S[i] * exp_qt * cdf_d1
            ) / 365.0
        else:
            theta[i] = (
                common_theta
                + r[i] * K[i] * exp_rt * (1.0 - cdf_d2)
                - q[i] * S[i] * exp_qt * (1.0 - cdf_d1)
            ) / 365.0

        # Rho
        if is_call[i]:
            rho[i] = (K[i] * Ti * exp_rt * cdf_d2) * 0.01
        else:
            rho[i] = (-K[i] * Ti * exp_rt * (1.0 - cdf_d2)) * 0.01

    return delta, gamma, vega, theta, rho
