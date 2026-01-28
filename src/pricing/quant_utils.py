"""
Highly Optimized Quantitative Implementation Utilities

This module provides low-level performance-critical utilities for numerical
finance calculations, including fast initial guesses and JIT-compiled math.
"""

import math
from typing import Tuple, Optional, Union

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


@njit(parallel=True, cache=True, fastmath=True)
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
    Optimized with fastmath and parallel execution.
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


@njit(cache=True, fastmath=True)
def calculate_d1_d2_jit(
    S: float, K: float, T: float, sigma: float, r: float, q: float
) -> Tuple[float, float]:
    """JIT compiled Black-Scholes d1 and d2 with fastmath."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


@njit(parallel=True, cache=True, fastmath=True)
def batch_bs_price_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    JIT compiled batch pricing for options.
    Supports 'out' parameter to minimize allocations.
    """
    n = len(S)
    if out is None:
        prices = np.empty(n, dtype=np.float64)
    else:
        prices = out

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


@njit(parallel=True, cache=True, fastmath=True)
def batch_greeks_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    out_delta: Optional[np.ndarray] = None,
    out_gamma: Optional[np.ndarray] = None,
    out_vega: Optional[np.ndarray] = None,
    out_theta: Optional[np.ndarray] = None,
    out_rho: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT compiled batch greeks calculation.
    Supports 'out' parameters to minimize allocations.
    """
    n = len(S)
    delta = out_delta if out_delta is not None else np.empty(n, dtype=np.float64)
    gamma = out_gamma if out_gamma is not None else np.empty(n, dtype=np.float64)
    vega = out_vega if out_vega is not None else np.empty(n, dtype=np.float64)
    theta = out_theta if out_theta is not None else np.empty(n, dtype=np.float64)
    rho = out_rho if out_rho is not None else np.empty(n, dtype=np.float64)

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

@njit(cache=True, fastmath=True)
def thomas_algorithm(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solves tridiagonal system Ax = rhs using Thomas algorithm (TDMA).
    A is defined by lower, diag, upper diagonals.
    Optimized with fastmath.
    Warning: This implementation modifies the inputs in-place for performance.
    """
    n = len(diag)
    # Forward elimination
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    
    for i in range(1, n-1):
        temp = diag[i] - lower[i-1] * c_prime[i-1]
        c_prime[i] = upper[i] / temp
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / temp
        
    d_prime[n-1] = (rhs[n-1] - lower[n-2] * d_prime[n-2]) / (diag[n-1] - lower[n-2] * c_prime[n-2])
    
    # Backward substitution
    x = np.zeros(n, dtype=np.float64)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

@njit(cache=True, fastmath=True, parallel=True)
def jit_cn_solver(
    s_grid: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float,
    is_call: bool,
    n_time: int
) -> np.ndarray:
    """
    JIT-compiled Crank-Nicolson solver loop with fastmath and parallel initialization.
    Optimized to minimize memory allocations by pre-allocating buffers.
    """
    M = len(s_grid) - 1 # Number of intervals, so M+1 points
    dt = maturity / n_time
    dS = s_grid[1] - s_grid[0] # Assumes uniform grid
    
    # Initial Condition (Terminal Payoff) - Parallelized
    V = np.empty(M + 1, dtype=np.float64)
    for i in prange(M + 1):
        if is_call:
            V[i] = max(s_grid[i] - strike, 0.0)
        else:
            V[i] = max(strike - s_grid[i], 0.0)
            
    # Coefficients that are constant in time
    sig2 = volatility**2
    mu = rate - dividend
    
    # Pre-calculate matrix coefficients for internal nodes
    indices = np.arange(1, M)
    S_i = s_grid[indices]
    
    alpha = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) - mu * S_i / dS)
    beta = -0.5 * dt * (sig2 * (S_i**2) / (dS**2) + rate)
    gamma = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) + mu * S_i / dS)
    
    diag_A = 1.0 - beta
    diag_B = 1.0 + beta
    
    # PRE-ALLOCATION: Move buffer creation out of the loop
    b = np.empty(M - 1, dtype=np.float64)
    lower_buf = -alpha[1:].copy()
    diag_buf = diag_A.copy()
    upper_buf = -gamma[:-1].copy()
    
    # Time stepping (Sequential outer loop, parallelized inner loops)
    for n in range(n_time, 0, -1):
        tau = (n_time - n + 1) * dt
        
        # Boundary Conditions
        if is_call:
            v_min_next = 0.0
            v_max_next = s_grid[M] - strike * math.exp(-rate * tau)
        else:
            v_min_next = strike * math.exp(-rate * tau)
            v_max_next = 0.0
            
        # Construct RHS vector 'b' without allocations - Parallelized
        for i in prange(M-1):
            b[i] = alpha[i] * V[i] + diag_B[i] * V[i+1] + gamma[i] * V[i+2]
            
        # Corrections from Implicit A
        b[0] += alpha[0] * v_min_next
        b[-1] += gamma[-1] * v_max_next
        
        # Reset buffers for Thomas algorithm (modifies in-place)
        lower_buf[:] = -alpha[1:]
        diag_buf[:] = diag_A
        upper_buf[:] = -gamma[:-1]
        
        V_new_internal = thomas_algorithm(
            lower_buf, 
            diag_buf, 
            upper_buf, 
            b
        )
        
        V[1:M] = V_new_internal
        V[0] = v_min_next
        V[M] = v_max_next
        
    return V

@njit(cache=True, fastmath=True)
def _laguerre_basis_jit(x: np.ndarray, degree: int) -> np.ndarray:
    n = len(x)
    basis = np.empty((n, degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = x
    if degree >= 2:
        for i in range(n):
            basis[i, 2] = x[i]**2 - 1.0
    if degree >= 3:
        for i in range(n):
            basis[i, 3] = x[i]**3 - 3.0 * x[i]
    return basis

@njit(parallel=True, cache=True, fastmath=True)
def jit_mc_european_price(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    q: float, 
    n_paths: int, 
    is_call: bool,
    antithetic: bool
) -> Tuple[float, float]:
    """
    JIT-optimized Monte Carlo for European options.
    Returns (price, standard_error).
    """
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)
    exp_rt = math.exp(-r * T)
    
    sum_payoff = 0.0
    sum_sq_payoff = 0.0
    
    # Process in chunks if n_paths is very large to avoid huge random array
    # But for now, we'll use Numba's parallel prange which is very efficient.
    
    actual_paths = n_paths
    if antithetic:
        loop_paths = n_paths // 2
    else:
        loop_paths = n_paths

    for i in prange(loop_paths):
        z = np.random.standard_normal()
        
        # Path 1
        st1 = S0 * math.exp(drift + diffusion * z)
        if is_call:
            payoff1 = max(st1 - K, 0.0)
        else:
            payoff1 = max(K - st1, 0.0)
        
        p1 = payoff1 * exp_rt
        
        if antithetic:
            # Path 2
            st2 = S0 * math.exp(drift - diffusion * z)
            if is_call:
                payoff2 = max(st2 - K, 0.0)
            else:
                payoff2 = max(K - st2, 0.0)
            p2 = payoff2 * exp_rt
            
            avg_p = (p1 + p2) / 2.0
            sum_payoff += avg_p * 2.0 # Total sum across all paths
            sum_sq_payoff += p1**2 + p2**2
        else:
            sum_payoff += p1
            sum_sq_payoff += p1**2
            
    price = sum_payoff / n_paths
    # Standard error calculation
    variance = (sum_sq_payoff / n_paths) - price**2
    std_err = math.sqrt(max(variance, 0.0) / n_paths)
    
    return price, std_err

@njit(parallel=True, cache=True, fastmath=True)
def jit_mc_european_price(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    q: float, 
    n_paths: int, 
    is_call: bool,
    antithetic: bool
) -> Tuple[float, float]:
    """
    JIT-optimized Monte Carlo for European options.
    Returns (price, standard_error).
    """
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)
    exp_rt = math.exp(-r * T)
    
    sum_payoff = 0.0
    sum_sq_payoff = 0.0
    
    actual_paths = n_paths
    if antithetic:
        loop_paths = n_paths // 2
    else:
        loop_paths = n_paths

    for i in prange(loop_paths):
        z = np.random.standard_normal()
        
        # Path 1
        st1 = S0 * math.exp(drift + diffusion * z)
        if is_call:
            payoff1 = max(st1 - K, 0.0)
        else:
            payoff1 = max(K - st1, 0.0)
        
        p1 = payoff1 * exp_rt
        
        if antithetic:
            # Path 2
            st2 = S0 * math.exp(drift - diffusion * z)
            if is_call:
                payoff2 = max(st2 - K, 0.0)
            else:
                payoff2 = max(K - st2, 0.0)
            p2 = payoff2 * exp_rt
            
            avg_p = (p1 + p2) / 2.0
            sum_payoff += avg_p * 2.0 # Total sum across all paths
            sum_sq_payoff += p1**2 + p2**2
        else:
            sum_payoff += p1
            sum_sq_payoff += p1**2
            
    price = sum_payoff / n_paths
    # Standard error calculation
    variance = (sum_sq_payoff / n_paths) - price**2
    std_err = math.sqrt(max(variance, 0.0) / n_paths)
    
    return price, std_err

@njit(cache=True, fastmath=True)
def jit_lsm_american(
    S0: float, K: float, T: float, r: float, sigma: float, q: float, 
    n_paths: int, n_steps: int, is_call: bool
) -> float:
    """
    JIT-compiled Longstaff-Schwartz algorithm with fastmath.
    """
    dt = T / n_steps
    df = math.exp(-r * dt)
    
    S = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    S[0, :] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        for i in range(n_paths):
            S[t, i] = S[t-1, i] * math.exp((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z[i])
            
    value = np.zeros(n_paths, dtype=np.float64)
    for i in range(n_paths):
        if is_call:
            value[i] = max(S[n_steps, i] - K, 0.0)
        else:
            value[i] = max(K - S[n_steps, i], 0.0)
            
    for t in range(n_steps - 1, 0, -1):
        itm_indices = []
        for i in range(n_paths):
            p = max(S[t, i] - K, 0.0) if is_call else max(K - S[t, i], 0.0)
            if p > 0:
                itm_indices.append(i)
        
        if len(itm_indices) == 0:
            value *= df
            continue
            
        itm = np.array(itm_indices)
        X = S[t, itm]
        Y = value[itm] * df
        
        basis = _laguerre_basis_jit(X / S0, 3)
        coeffs = np.linalg.solve(basis.T @ basis, basis.T @ Y)
        continuation_value = basis @ coeffs
        
        current_payoff = np.zeros(n_paths, dtype=np.float64)
        for i in itm:
            current_payoff[i] = max(S[t, i] - K, 0.0) if is_call else max(K - S[t, i], 0.0)

        for k in range(len(itm)):
            idx = itm[k]
            if current_payoff[idx] > continuation_value[k]:
                value[idx] = current_payoff[idx]
            else:
                value[idx] *= df
        
        mask = np.ones(n_paths, dtype=np.bool_)
        mask[itm] = False
        value[mask] *= df
        
    return float(np.mean(value * df))

