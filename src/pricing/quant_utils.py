"""
🚀 APOTHEOSIS: High-Performance Quantitative Kernels
Targets: Pure NumPy, Vectorized

Performance Notes:
- Optimized for vectorized NumPy operations.
- Parallelism provided by NumPy's underlying BLAS/LAPACK.
"""

import numpy as np
from scipy.special import erf

from src.shared.math_utils import (
    calculate_d1_d2,
    calculate_d1_d2_scalar,
)


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
    Vectorized with NumPy.
    """
    X = strike * np.exp(-rate * maturity)
    val = 2.5066282746310005 / (np.sqrt(maturity) * (spot + X))

    exp_qt = np.exp(-dividend * maturity)
    intrinsic = np.where(option_type == 0,
                         np.maximum(spot * exp_qt - X, 0.0),
                         np.maximum(X - spot * exp_qt, 0.0))

    term = market_price - intrinsic / 2.0
    sigma = val * (term + np.sqrt(np.maximum(term**2 - intrinsic**2 / np.pi, 0.0)))

    return np.clip(sigma, 0.001, 5.0)


def batch_bs_price_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    out: np.ndarray | None = None
) -> np.ndarray:
    """
    Batch pricing for options using NumPy.
    """
    if out is None:
        prices = np.empty(S.shape, dtype=np.float64)
    else:
        prices = out

    t_mask = T < 1e-7
    prices[t_mask] = np.where(is_call[t_mask],
                              np.maximum(S[t_mask] - K[t_mask], 0.0),
                              np.maximum(K[t_mask] - S[t_mask], 0.0))

    not_t_mask = ~t_mask
    if np.any(not_t_mask):
        S_n = S[not_t_mask]
        K_n = K[not_t_mask]
        T_n = T[not_t_mask]
        sig_n = sigma[not_t_mask]
        r_n = r[not_t_mask]
        q_n = q[not_t_mask]
        is_call_n = is_call[not_t_mask]

        sig_sqrt_t = sig_n * np.sqrt(T_n)
        d1 = (np.log(S_n / K_n) + (r_n - q_n + 0.5 * sig_n**2) * T_n) / sig_sqrt_t
        d2 = d1 - sig_sqrt_t

        nd1 = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + erf(d2 / np.sqrt(2.0)))

        exp_qt = np.exp(-q_n * T_n)
        exp_rt = np.exp(-r_n * T_n)

        p_call = np.maximum(S_n * exp_qt * nd1 - K_n * exp_rt * nd2, 0.0)
        p_put = np.maximum(K_n * exp_rt * (1.0 - nd2) - S_n * exp_qt * (1.0 - nd1), 0.0)

        prices[not_t_mask] = np.where(is_call_n, p_call, p_put)

    return prices


def batch_greeks_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    is_call: np.ndarray,
    out_delta: np.ndarray | None = None,
    out_gamma: np.ndarray | None = None,
    out_vega: np.ndarray | None = None,
    out_theta: np.ndarray | None = None,
    out_rho: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch greeks calculation using NumPy.
    """
    Ti = np.maximum(T, 1e-7)
    sqrt_T = np.sqrt(Ti)
    d1, d2 = calculate_d1_d2(S, K, Ti, sigma, r, q)

    inv_sqrt_2pi = 1.0 / 2.5066282746310005
    pdf_d1 = np.exp(-0.5 * d1**2) * inv_sqrt_2pi
    cdf_d1 = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0)))
    cdf_d2 = 0.5 * (1.0 + erf(d2 / np.sqrt(2.0)))

    exp_qt = np.exp(-q * Ti)
    exp_rt = np.exp(-r * Ti)

    is_c = is_call.astype(np.float64)
    is_p = 1.0 - is_c

    delta = is_c * (exp_qt * cdf_d1) + is_p * (exp_qt * (cdf_d1 - 1.0))
    gamma = (exp_qt * pdf_d1) / (S * sigma * sqrt_T)
    vega = (S * exp_qt * pdf_d1 * sqrt_T) * 0.01

    common_theta = -(S * pdf_d1 * sigma * exp_qt) / (2 * sqrt_T)
    call_theta = (common_theta - r * K * exp_rt * cdf_d2 + q * S * exp_qt * cdf_d1) / 365.0
    put_theta = (common_theta + r * K * exp_rt * (1.0 - cdf_d2) - q * S * exp_qt * (1.0 - cdf_d1)) / 365.0
    theta = is_c * call_theta + is_p * put_theta

    call_rho = (K * Ti * exp_rt * cdf_d2) * 0.01
    put_rho = (-K * Ti * exp_rt * (1.0 - cdf_d2)) * 0.01
    rho = is_c * call_rho + is_p * put_rho

    if out_delta is not None:
        out_delta[:] = delta
    if out_gamma is not None:
        out_gamma[:] = gamma
    if out_vega is not None:
        out_vega[:] = vega
    if out_theta is not None:
        out_theta[:] = theta
    if out_rho is not None:
        out_rho[:] = rho

    return delta, gamma, vega, theta, rho

def thomas_algorithm(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves tridiagonal system Ax = rhs using Thomas algorithm."""
    n = len(diag)
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    
    for i in range(1, n-1):
        temp = diag[i] - lower[i-1] * c_prime[i-1]
        c_prime[i] = upper[i] / temp
        d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / temp
        
    d_prime[n-1] = (rhs[n-1] - lower[n-2] * d_prime[n-2]) / (diag[n-1] - lower[n-2] * c_prime[n-2])
    
    x = np.zeros(n, dtype=np.float64)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

def jit_cn_solver(
    s_grid: np.ndarray, strike: float, maturity: float, rate: float, 
    volatility: float, dividend: float, is_call: bool, n_time: int
) -> np.ndarray:
    """Crank-Nicolson solver using NumPy."""
    M = len(s_grid) - 1
    dt = maturity / n_time
    dS = s_grid[1] - s_grid[0]
    
    V = np.where(is_call, np.maximum(s_grid - strike, 0.0), np.maximum(strike - s_grid, 0.0))
            
    sig2 = volatility**2
    mu = rate - dividend
    indices = np.arange(1, M)
    S_i = s_grid[indices]
    
    alpha = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) - mu * S_i / dS)
    beta = -0.5 * dt * (sig2 * (S_i**2) / (dS**2) + rate)
    gamma = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) + mu * S_i / dS)
    
    diag_A = 1.0 - beta
    diag_B = 1.0 + beta
    
    for n in range(n_time, 0, -1):
        tau = (n_time - n + 1) * dt
        if is_call:
            v_min_next = 0.0
            v_max_next = s_grid[M] - strike * np.exp(-rate * tau)
        else:
            v_min_next = strike * np.exp(-rate * tau)
            v_max_next = 0.0
            
        b = alpha * V[:-2] + diag_B * V[1:-1] + gamma * V[2:]
        b[0] += alpha[0] * v_min_next
        b[-1] += gamma[-1] * v_max_next
        
        lower_buf = -alpha[1:]
        diag_buf = diag_A
        upper_buf = -gamma[:-1]
        
        V_new_internal = thomas_algorithm(lower_buf, diag_buf, upper_buf, b)
        V[1:M] = V_new_internal
        V[0] = v_min_next
        V[M] = v_max_next
        
    return V

def vectorized_newton_raphson_iv_jit(
    market_prices: np.ndarray, spots: np.ndarray, strikes: np.ndarray,
    maturities: np.ndarray, rates: np.ndarray, dividends: np.ndarray,
    is_call: np.ndarray, sigma: np.ndarray, tolerance: float = 1e-8, max_iterations: int = 100
) -> np.ndarray:
    """Newton-Raphson loop for IV recovery using NumPy."""
    sigma = sigma.copy()
    active = np.ones(len(market_prices), dtype=bool)
    inv_sqrt_2pi = 1.0 / 2.5066282746310005

    for _ in range(max_iterations):
        if not np.any(active):
            break
            
        S, K, T, r, q = spots[active], strikes[active], maturities[active], rates[active], dividends[active]
        sig = sigma[active]
        Ti = np.maximum(T, 1e-7)
        sqrt_T = np.sqrt(Ti)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * Ti) / (sig * sqrt_T)
        d2 = d1 - sig * sqrt_T
        nd1 = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + erf(d2 / np.sqrt(2.0)))
        
        exp_qt = np.exp(-q * Ti)
        exp_rt = np.exp(-r * Ti)
        
        price = np.where(is_call[active],
                         S * exp_qt * nd1 - K * exp_rt * nd2,
                         K * exp_rt * (1.0 - nd2) - S * exp_qt * (1.0 - nd1))
        
        pdf_d1 = np.exp(-0.5 * d1**2) * inv_sqrt_2pi
        vega = S * exp_qt * pdf_d1 * sqrt_T
        
        diff = price - market_prices[active]
        
        # Check tolerance
        newly_inactive = np.abs(diff) < tolerance
        active_indices = np.where(active)[0]
        active[active_indices[newly_inactive]] = False
        
        if np.any(~newly_inactive):
            sigma[active] -= np.clip(diff[~newly_inactive] / np.maximum(vega[~newly_inactive], 1e-12), -0.5, 0.5)
            sigma[active] = np.clip(sigma[active], 1e-4, 5.0)
            
    return sigma

def heston_char_func_jit(u, T, r, v0, kappa, theta, sigma, rho) -> complex:
    """Heston characteristic function using NumPy."""
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g = (xi + d) / (xi - d)
    exp_dT = np.exp(d * T)
    A = (kappa * theta / sigma**2) * ((xi + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
    B = (v0 / sigma**2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(A + B)

def jit_mc_european_price(S0, K, T, r, sigma, q, n_paths, is_call, antithetic, z_innovations=None):
    """Monte Carlo for European options using NumPy."""
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    exp_rt = np.exp(-r * T)
    
    actual_paths = n_paths // 2 if antithetic else n_paths
    if z_innovations is not None:
        z = z_innovations
    else:
        z = np.random.standard_normal(actual_paths)
        
    st1 = S0 * np.exp(drift + diffusion * z)
    p1 = np.where(is_call, np.maximum(st1 - K, 0.0), np.maximum(K - st1, 0.0)) * exp_rt
    
    if antithetic:
        st2 = S0 * np.exp(drift - diffusion * z)
        p2 = np.where(is_call, np.maximum(st2 - K, 0.0), np.maximum(K - st2, 0.0)) * exp_rt
        combined = np.concatenate([p1, p2])
    else:
        combined = p1
        
    price = np.mean(combined)
    std_err = np.sqrt(np.maximum(np.var(combined) / n_paths, 0.0))
    return price, std_err

def jit_mc_european_price_and_greeks(S0, K, T, r, sigma, q, n_paths, is_call, antithetic):
    """Pathwise Sensitivity (PWM) Monte Carlo using NumPy."""
    drift_part = (r - q - 0.5 * sigma**2) * T
    diffusion_part = sigma * np.sqrt(T)
    sqrt_T, exp_rt = np.sqrt(T), np.exp(-r * T)
    
    actual_paths = n_paths // 2 if antithetic else n_paths
    z = np.random.standard_normal(actual_paths)
    
    def calc_stats(z_val):
        st = S0 * np.exp(drift_part + diffusion_part * z_val)
        payoff = np.where(is_call, np.maximum(st - K, 0.0), np.maximum(K - st, 0.0)) * exp_rt
        ind = np.where(is_call, (st > K).astype(float), (st < K).astype(float))
        delta = exp_rt * ind * (st / S0)
        vega = exp_rt * ind * st * (z_val * sqrt_T - sigma * T) * 0.01
        rho = (-T * payoff + exp_rt * ind * st * T) * 0.01
        return payoff, delta, vega, rho

    p1, d1, v1, r1 = calc_stats(z)
    
    if antithetic:
        p2, d2, v2, r2 = calc_stats(-z)
        p, d, v, rho = (p1+p2)/2, (d1+d2)/2, (v1+v2)/2, (r1+r2)/2
    else:
        p, d, v, rho = p1, d1, v1, r1
        
    return np.mean(p), np.mean(d), 0.0, np.mean(v), np.mean(rho)

def jit_mc_european_with_control_variate(S0, K, T, r, sigma, q, n_paths, is_call, antithetic, z_innovations=None):
    """MC with Control Variate using NumPy."""
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    exp_rt = np.exp(-r * T)
    expected_st = S0 * np.exp((r - q) * T)
    
    actual_paths = n_paths // 2 if antithetic else n_paths
    if z_innovations is not None:
        z = z_innovations
    else:
        z = np.random.standard_normal(actual_paths)
        
    st1 = S0 * np.exp(drift + diffusion * z)
    p1 = np.where(is_call, np.maximum(st1 - K, 0.0), np.maximum(K - st1, 0.0)) * exp_rt
    
    if antithetic:
        st2 = S0 * np.exp(drift - diffusion * z)
        p2 = np.where(is_call, np.maximum(st2 - K, 0.0), np.maximum(K - st2, 0.0)) * exp_rt
        p_all = np.concatenate([p1, p2])
        st_all = np.concatenate([st1, st2])
    else:
        p_all, st_all = p1, st1
        
    mu_p, mu_s = np.mean(p_all), np.mean(st_all)
    cov_ps = np.mean(p_all * st_all) - mu_p * mu_s
    var_s = np.var(st_all)
    
    beta = cov_ps / max(var_s, 1e-12)
    price_cv = mu_p - beta * (mu_s - expected_st)
    
    var_p = np.var(p_all)
    r2 = (cov_ps**2) / (max(var_p * var_s, 1e-12))
    var_cv = var_p * (1.0 - min(r2, 0.999))
    
    return price_cv, np.sqrt(np.maximum(var_cv / n_paths, 0.0))

def gpu_mc_european_price(S0, K, T, r, sigma, q, n_paths, is_call, antithetic):
    """GPU-accelerated MC using CuPy (optional)."""
    try:
        import cupy as cp
    except ImportError:
        return None
        
    drift, diffusion, exp_rt = (r - q - 0.5 * sigma**2) * T, sigma * np.sqrt(T), np.exp(-r * T)
    actual_paths = n_paths // 2 if antithetic else n_paths
    z = cp.random.standard_normal(actual_paths, dtype=cp.float32)
    st1 = S0 * cp.exp(drift + diffusion * z)
    p1 = cp.where(is_call, cp.maximum(st1 - K, 0.0), cp.maximum(K - st1, 0.0)) * exp_rt
    if antithetic:
        st2 = S0 * cp.exp(drift - diffusion * z)
        p2 = cp.where(is_call, cp.maximum(st2 - K, 0.0), cp.maximum(K - st2, 0.0)) * exp_rt
        combined = cp.concatenate([p1, p2])
    else: 
        combined = p1
    return float(cp.mean(combined)), float(1.96 * np.sqrt(float(cp.var(combined)) / n_paths))

def jit_generate_log_paths(S0, T, r, sigma, q, n_paths, n_steps):
    """Generate log-paths: log(S_t)."""
    dt = T / n_steps
    drift, diffusion = (r - q - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt)
    Z = np.random.standard_normal((n_steps, n_paths))
    log_returns = drift + diffusion * Z
    log_paths = np.zeros((n_steps + 1, n_paths))
    log_paths[1:, :] = np.cumsum(log_returns, axis=0)
    return log_paths + np.log(S0)

def jit_generate_paths(S0, T, r, sigma, q, n_paths, n_steps):
    """Highly optimized vectorized path generation using NumPy."""
    log_paths = jit_generate_log_paths(S0, T, r, sigma, q, n_paths, n_steps)
    return np.exp(log_paths).T

def _laguerre_basis_jit(x, degree):
    """Generate Laguerre basis functions using NumPy."""
    n = len(x)
    basis = np.ones((n, degree + 1), dtype=np.float64)
    if degree >= 1:
        basis[:, 1] = 1.0 - x
    if degree >= 2:
        basis[:, 2] = 0.5 * (2.0 - 4.0 * x + x**2)
    if degree >= 3:
        basis[:, 3] = (1.0 / 6.0) * (6.0 - 18.0 * x + 9.0 * x**2 - x**3)
    return basis

def _jit_solve_normal_equations(X, y):
    """Normal Equations solver using NumPy."""
    return np.linalg.solve(X.T @ X, X.T @ y)

def jit_lsm_american(S0, K, T, r, sigma, q, n_paths, n_steps, is_call):
    """LSM algorithm using NumPy."""
    dt = T / n_steps
    df = np.exp(-r * dt)
    
    # Optimized path generation
    paths = jit_generate_paths(S0, T, r, sigma, q, n_paths, n_steps) # (n_paths, n_steps + 1)
    S = paths.T # (n_steps + 1, n_paths)
        
    value = np.where(is_call, np.maximum(S[n_steps, :] - K, 0.0), np.maximum(K - S[n_steps, :], 0.0))
    
    for t in range(n_steps - 1, 0, -1):
        payoff_t = np.where(is_call, np.maximum(S[t, :] - K, 0.0), np.maximum(K - S[t, :], 0.0))
        itm_mask = payoff_t > 0
        if not np.any(itm_mask):
            value *= df
            continue
            
        X_itm, Y_itm = S[t, itm_mask], value[itm_mask] * df
        basis = _laguerre_basis_jit(X_itm / S0, 3)
        continuation_value = basis @ _jit_solve_normal_equations(basis, Y_itm)
        
        exercise = payoff_t[itm_mask] > continuation_value
        
        # Get indices of ITM paths
        itm_indices = np.where(itm_mask)[0]
        
        # Update values
        value[itm_indices[exercise]] = payoff_t[itm_indices[exercise]]
        value[itm_indices[~exercise]] *= df
        value[~itm_mask] *= df
        
    return np.mean(value) * df

def scalar_bs_price_jit(S, K, T, sigma, r, q, is_call):
    """Scalar BS pricing using NumPy."""
    if T < 1e-7:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1, d2 = calculate_d1_d2_scalar(S, K, T, sigma, r, q)
    nd1, nd2 = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0))), 0.5 * (1.0 + erf(d2 / np.sqrt(2.0)))
    price = S * np.exp(-q * T) * nd1 - K * np.exp(-r * T) * nd2 if is_call else K * np.exp(-r * T) * (1.0 - nd2) - S * np.exp(-q * T) * (1.0 - nd1)
    return max(price, 0.0)

def scalar_greeks_jit(S, K, T, sigma, r, q, is_call):
    """Scalar Greeks calculation using NumPy."""
    Ti = max(T, 1e-7)
    sqrt_T = np.sqrt(Ti)
    d1, d2 = calculate_d1_d2_scalar(S, K, Ti, sigma, r, q)
    inv_sqrt_2pi = 1.0 / 2.5066282746310005
    pdf_d1 = np.exp(-0.5 * d1**2) * inv_sqrt_2pi
    cdf_d1, cdf_d2 = 0.5 * (1.0 + erf(d1 / np.sqrt(2.0))), 0.5 * (1.0 + erf(d2 / np.sqrt(2.0)))
    exp_qt, exp_rt = np.exp(-q * Ti), np.exp(-r * Ti)
    if is_call:
        delta, rho = exp_qt * cdf_d1, (K * Ti * exp_rt * cdf_d2) * 0.01
        theta = (-(S * pdf_d1 * sigma * exp_qt) / (2 * sqrt_T) - r * K * exp_rt * cdf_d2 + q * S * exp_qt * cdf_d1) / 365.0
    else:
        delta, rho = exp_qt * (cdf_d1 - 1.0), (-K * Ti * exp_rt * (1.0 - cdf_d2)) * 0.01
        theta = (-(S * pdf_d1 * sigma * exp_qt) / (2 * sqrt_T) + r * K * exp_rt * (1.0 - cdf_d2) - q * S * exp_qt * (1.0 - cdf_d1)) / 365.0
    return delta, (exp_qt * pdf_d1) / (S * sigma * sqrt_T), (S * exp_qt * pdf_d1 * sqrt_T) * 0.01, theta, rho

def warmup_jit():
    """Dummy warmup for compatibility (not needed for Pure NumPy)."""
    pass
