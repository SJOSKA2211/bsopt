"""
 APOTHEOSIS: High-Performance Quantitative Kernels
Targets: Numba JIT, Parallel, Vectorized
"""

import numpy as np
from numba import njit

# Scheme Constants
SCHEME_EULER = 0
SCHEME_MILSTEIN = 1
SCHEME_EULER_MULTI = 2


@njit
def fast_normal_cdf_v2(x):
    """Rational approximation of CDF."""
    INV_SQRT2 = 0.7071067811865476
    P = 0.3275911
    A1, A2, A3, A4, A5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    abs_x = np.abs(x) * INV_SQRT2
    t = 1.0 / (1.0 + P * abs_x)
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    y = 1.0 - poly * np.exp(-abs_x * abs_x)
    return 0.5 * (1.0 + np.sign(x) * y)


def generate_log_paths_v2(S0, T, r, sigma, q, n_paths, n_steps):
    """Generate log-paths (Non-JIT)."""
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.standard_normal((n_steps, n_paths))
    log_paths = np.zeros((n_steps + 1, n_paths))
    log_paths[0, :] = np.log(S0)
    for i in range(n_steps):
        log_paths[i + 1, :] = log_paths[i, :] + drift + diffusion * Z[i, :]
    return log_paths


def generate_paths_v2(S0, T, r, sigma, q, n_paths, n_steps, scheme=SCHEME_EULER):
    """Optimized path generation."""
    if scheme == SCHEME_MILSTEIN:
        dt = T / n_steps
        mu = r - q
        S = np.full((n_paths, n_steps + 1), S0, dtype=np.float64)
        Z = np.random.standard_normal((n_paths, n_steps))
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            dW = Z[:, t] * sqrt_dt
            S[:, t + 1] = S[:, t] * (1 + mu * dt + sigma * dW + 0.5 * (sigma**2) * (dW**2 - dt))
        return S

    log_paths = generate_log_paths_v2(S0, T, r, sigma, q, n_paths, n_steps)
    return np.exp(log_paths).T


def fused_arithmetic_asian_payoff_v2(log_paths, K, r, T, is_call, is_fixed):
    """Fused kernel (Non-JIT)."""
    n_steps_p1, n_paths = log_paths.shape
    exp_rt = np.exp(-r * T)
    prices = np.exp(log_paths[1:, :])
    arith_means = np.mean(prices, axis=0)
    if is_fixed:
        payoffs = arith_means - K if is_call else K - arith_means
    else:
        payoffs = (
            np.exp(log_paths[-1, :]) - arith_means
            if is_call
            else arith_means - np.exp(log_paths[-1, :])
        )
    return np.maximum(payoffs, 0.0) * exp_rt


def fused_lookback_payoff_v2(log_paths, K, r, T, is_call, is_floating):
    """Fused kernel (Non-JIT)."""
    exp_rt = np.exp(-r * T)
    if is_floating:
        extrema = np.min(log_paths, axis=0) if is_call else np.max(log_paths, axis=0)
        payoffs = (
            np.exp(log_paths[-1, :]) - np.exp(extrema)
            if is_call
            else np.exp(extrema) - np.exp(log_paths[-1, :])
        )
    else:
        extrema = np.max(log_paths, axis=0) if is_call else np.min(log_paths, axis=0)
        payoffs = np.exp(extrema) - K if is_call else K - np.exp(extrema)
    return np.maximum(payoffs, 0.0) * exp_rt


@njit
def batch_bs_price_jit_v2(S, K, T, sigma, r, q, is_call):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    exp_rt, exp_qt = np.exp(-r * T), np.exp(-q * T)
    from scipy.stats import norm

    if isinstance(is_call, bool):
        return S * exp_qt * norm.cdf(d1 if is_call else -d1) - K * exp_rt * norm.cdf(
            d2 if is_call else -d2
        )
    res = np.empty_like(S)
    res[is_call] = S[is_call] * exp_qt[is_call] * norm.cdf(d1[is_call]) - K[is_call] * exp_rt[
        is_call
    ] * norm.cdf(d2[is_call])
    res[~is_call] = K[~is_call] * exp_rt[~is_call] * norm.cdf(-d2[~is_call]) - S[~is_call] * exp_qt[
        ~is_call
    ] * norm.cdf(-d1[~is_call])
    return res


@njit
def batch_greeks_jit_v2(S, K, T, sigma, r, q, is_call):
    n = len(S)
    delta, gamma, theta, vega, rho = np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        d, g, th, v, rh = scalar_greeks_jit_v2(S[i], K[i], T[i], sigma[i], r[i], q[i], is_call[i])
        delta[i], gamma[i], theta[i], vega[i], rho[i] = d, g, th, v, rh
    return delta, gamma, vega, theta, rho


@njit
def scalar_greeks_jit_v2(S, K, T, sigma, r, q, is_call):
    from scipy.stats import norm

    Ti, sig = max(T, 1e-7), max(sigma, 1e-12)
    sqrt_T = np.sqrt(Ti)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * Ti) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    exp_qt, exp_rt = np.exp(-q * Ti), np.exp(-r * Ti)
    nd1, nd2, pdf_d1 = norm.cdf(d1), norm.cdf(d2), norm.pdf(d1)
    gamma = (exp_qt * pdf_d1) / (S * sig * sqrt_T)
    vega = (S * exp_qt * pdf_d1 * sqrt_T) * 0.01
    common_theta = -(S * pdf_d1 * sig * exp_qt) / (2 * sqrt_T)
    if is_call:
        delta, rho = exp_qt * nd1, (K * Ti * exp_rt * nd2) * 0.01
        theta = (common_theta - r * K * exp_rt * nd2 + q * S * exp_qt * nd1) / 365.0
    else:
        delta, rho = exp_qt * (nd1 - 1.0), (-K * Ti * exp_rt * (1.0 - nd2)) * 0.01
        theta = (common_theta + r * K * exp_rt * (1.0 - nd2) - q * S * exp_qt * (1.0 - nd1)) / 365.0
    return delta, gamma, theta, vega, rho


@njit
def corrado_miller_initial_guess(market_price, spot, strike, maturity, rate, dividend, is_call):
    n = len(market_price)
    sigma = np.empty(n)
    FACTOR = 2.5066282746310005
    for i in range(n):
        X = strike[i] * np.exp(-rate[i] * maturity[i])
        val = FACTOR / (np.sqrt(maturity[i]) * (spot[i] + X))
        exp_qt = np.exp(-dividend[i] * maturity[i])
        intrinsic = max(spot[i] * exp_qt - X, 0.0) if is_call[i] else max(X - spot[i] * exp_qt, 0.0)
        term = market_price[i] - intrinsic / 2.0
        sigma[i] = val * (term + np.sqrt(max(term**2 - intrinsic**2 / np.pi, 0.0)))
    return np.clip(sigma, 0.001, 5.0)


@njit
def heston_char_func_jit(u, T, r, v0, kappa, theta, sigma, rho) -> complex:
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g = (xi + d) / (xi - d)
    exp_dT = np.exp(d * T)
    A = (kappa * theta / sigma**2) * ((xi + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g)))
    B = (v0 / sigma**2) * (xi + d) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(A + B)


@njit
def jit_mc_european_price_v2(
    S0, K, T, r, sigma, q, n_paths, is_call, antithetic, z_innovations=None, scheme=SCHEME_EULER
):
    actual_paths = n_paths // 2 if antithetic else n_paths
    exp_rt = np.exp(-r * T)
    drift, diffusion = (r - q - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    z = z_innovations if z_innovations is not None else np.random.standard_normal(actual_paths)
    if antithetic:
        s1, s2 = S0 * np.exp(drift + diffusion * z), S0 * np.exp(drift - diffusion * z)
        p1, p2 = (
            np.maximum(s1 - K if is_call else K - s1, 0.0),
            np.maximum(s2 - K if is_call else K - s2, 0.0),
        )
        payoffs = (p1 + p2) * 0.5 * exp_rt
    else:
        st = S0 * np.exp(drift + diffusion * z)
        payoffs = np.maximum(st - K if is_call else K - st, 0.0) * exp_rt
    return np.mean(payoffs), np.sqrt(max(np.var(payoffs) / n_paths, 0.0))


# ─── JIT Kernels & Quantitative Utilities ──────────────────────────────────────────


@njit
def scalar_bs_price_jit(S, K, T, sigma, r, q, is_call):
    """Scalar Black-Scholes price."""
    Ti, sig = max(T, 1e-7), max(sigma, 1e-12)
    vol_sqrt_t = sig * np.sqrt(Ti)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * Ti) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    exp_rt, exp_qt = np.exp(-r * Ti), np.exp(-q * Ti)
    from scipy.stats import norm

    if is_call:
        return S * exp_qt * norm.cdf(d1) - K * exp_rt * norm.cdf(d2)
    return K * exp_rt * norm.cdf(-d2) - S * exp_qt * norm.cdf(-d1)


@njit
def _laguerre_basis_jit(x, n):
    """Laguerre polynomial basis for LSM."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return np.exp(-x / 2)
    if n == 2:
        return np.exp(-x / 2) * (1 - x)
    if n == 3:
        return np.exp(-x / 2) * (1 - 2 * x + x**2 / 2)
    return np.exp(-x / 2)


@njit
def vectorized_newton_raphson_iv_jit(
    market_price, S, K, T, r, q, is_call, initial_guess=None, tol=1e-6, max_iter=100
):
    """Vectorized IV calculation using Newton-Raphson."""
    if initial_guess is not None:
        iv = initial_guess.copy()
    else:
        iv = corrado_miller_initial_guess(market_price, S, K, T, r, q, is_call)

    for _ in range(max_iter):
        p = batch_bs_price_jit_v2(S, K, T, iv, r, q, is_call)
        diff = p - market_price
        if np.all(np.abs(diff) < tol):
            break
        _, _, _, vega, _ = batch_greeks_jit_v2(S, K, T, iv, r, q, is_call)
        iv -= diff / (vega * 100.0 + 1e-12)
    return np.clip(iv, 0.0001, 5.0)


@njit
def thomas_algorithm(a, b, c, d):
    """Solve tridiagonal system of linear equations."""
    n = len(d)
    c_new = np.zeros(n - 1)
    d_new = np.zeros(n)

    if n > 0:
        c_new[0] = c[0] / (b[0] + 1e-12)
        d_new[0] = d[0] / (b[0] + 1e-12)

        for i in range(1, n - 1):
            denom = b[i] - a[i - 1] * c_new[i - 1]
            c_new[i] = c[i] / (denom + 1e-12)
        for i in range(1, n):
            denom = b[i] - a[i - 1] * c_new[i - 1]
            d_new[i] = (d[i] - a[i - 1] * d_new[i - 1]) / (denom + 1e-12)

        x = np.zeros(n)
        x[-1] = d_new[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_new[i] - c_new[i] * x[i + 1]
        return x
    return np.zeros(0)


@njit
def jit_cn_solver(s_grid, K, T, r, sigma, q, is_call, N):
    """
    Crank-Nicolson solver for the Black-Scholes PDE.
    M: number of stock price steps (len(s_grid) - 1)
    N: number of time steps
    """
    M = len(s_grid) - 1
    dt = T / N
    dS = s_grid[1] - s_grid[0]

    # Initial condition (payoff at maturity)
    if is_call:
        V = np.maximum(s_grid - K, 0.0)
    else:
        V = np.maximum(K - s_grid, 0.0)

    # Pre-calculate coefficients
    # a_j, b_j, c_j for the tridiagonal matrix
    # Equation: V_j^{n} = a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1}
    
    # Grid indices 1 to M-1
    j = np.arange(1, M)
    sigma2 = sigma**2
    j2 = j**2
    
    alpha = 0.25 * dt * (sigma2 * j2 - (r - q) * j)
    beta = -0.5 * dt * (sigma2 * j2 + r)
    gamma = 0.25 * dt * (sigma2 * j2 + (r - q) * j)

    # Left-hand side tridiagonal matrix (A) coefficients
    # (1 - beta) V_j^n - alpha V_{j-1}^n - gamma V_{j+1}^n = (1 + beta) V_j^{n+1} + alpha V_{j-1}^{n+1} + gamma V_{j+1}^{n+1}
    a_lhs = -alpha
    b_lhs = 1.0 - beta
    c_lhs = -gamma
    
    # Right-hand side coefficients
    a_rhs = alpha
    b_rhs = 1.0 + beta
    c_rhs = gamma

    # Time stepping (backward in time)
    for n in range(N):
        # 1. Compute RHS: B * V^{n+1}
        rhs = np.zeros(M - 1)
        for i in range(M - 1):
            # Inner points
            val = b_rhs[i] * V[i+1] + a_rhs[i] * V[i] + c_rhs[i] * V[i+2]
            rhs[i] = val

        # 2. Apply Boundary Conditions to RHS
        # S=0 boundary (V_0)
        if is_call:
            v_0_new = 0.0
            v_M_new = s_grid[M] * np.exp(-q * (n+1) * dt) - K * np.exp(-r * (n+1) * dt)
        else:
            v_0_new = K * np.exp(-r * (n+1) * dt)
            v_M_new = 0.0
            
        rhs[0] += a_lhs[0] * v_0_new + a_rhs[0] * V[0]
        rhs[-1] += c_lhs[-1] * v_M_new + c_rhs[-1] * V[M]

        # 3. Solve Tridiagonal System A * V^n = RHS
        # thomas_algorithm(a, b, c, d)
        V[1:M] = thomas_algorithm(a_lhs[1:], b_lhs, c_lhs[:-1], rhs)
        
        # 4. Update Boundaries
        V[0] = v_0_new
        V[M] = v_M_new

    return V


# Backward compatibility stubs
def warmup_jit():
    pass


fast_normal_cdf = fast_normal_cdf_v2
jit_generate_log_paths = generate_log_paths_v2
jit_generate_paths = generate_paths_v2
fused_arithmetic_asian_payoff = fused_arithmetic_asian_payoff_v2
fused_lookback_payoff = fused_lookback_payoff_v2
batch_bs_price_jit = batch_bs_price_jit_v2
batch_greeks_jit = batch_greeks_jit_v2
scalar_greeks_jit = scalar_greeks_jit_v2
gpu_mc_european_price = jit_mc_european_price_v2
jit_mc_european_price = jit_mc_european_price_v2
jit_mc_european_price_and_greeks = None
jit_lsm_american = None
jit_mc_european_with_control_variate = None
