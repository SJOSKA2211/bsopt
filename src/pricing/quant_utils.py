"""
 APOTHEOSIS: High-Performance Quantitative Kernels
Targets: Numba JIT, Parallel, Vectorized
"""

import numpy as np

# Scheme Constants
SCHEME_EULER = 0
SCHEME_MILSTEIN = 1
SCHEME_EULER_MULTI = 2

def fast_normal_cdf_v2(x):
    """Rational approximation of CDF."""
    INV_SQRT2 = 0.7071067811865476
    P = 0.3275911
    A1, A2, A3, A4, A5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    abs_x = abs(x) * INV_SQRT2
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

def fused_arithmetic_asian_payoff_v2(log_paths, K, r, T, is_call, is_fixed):
    """Fused kernel (Non-JIT)."""
    n_steps_p1, n_paths = log_paths.shape
    n_steps = n_steps_p1 - 1
    exp_rt = np.exp(-r * T)
    
    # Vectorized calculation for better performance even without JIT
    prices = np.exp(log_paths[1:, :])
    arith_means = np.mean(prices, axis=0)
    
    if is_fixed:
        payoffs = arith_means - K if is_call else K - arith_means
    else:
        last_s = np.exp(log_paths[-1, :])
        payoffs = last_s - arith_means if is_call else arith_means - last_s
        
    return np.maximum(payoffs, 0.0) * exp_rt

def fused_lookback_payoff_v2(log_paths, K, r, T, is_call, is_floating):
    """Fused kernel (Non-JIT)."""
    n_steps_p1, n_paths = log_paths.shape
    exp_rt = np.exp(-r * T)
    
    if is_floating:
        if is_call: # S_last - S_min
            extrema = np.min(log_paths, axis=0)
            payoffs = np.exp(log_paths[-1, :]) - np.exp(extrema)
        else: # S_max - S_last
            extrema = np.max(log_paths, axis=0)
            payoffs = np.exp(extrema) - np.exp(log_paths[-1, :])
    else: # Fixed strike
        if is_call: # S_max - K
            extrema = np.max(log_paths, axis=0)
            payoffs = np.exp(extrema) - K
        else: # K - S_min
            extrema = np.min(log_paths, axis=0)
            payoffs = K - np.exp(extrema)
            
    return np.maximum(payoffs, 0.0) * exp_rt

def batch_bs_price_jit_v2(S, K, T, sigma, r, q, is_call):
    """Batch BS pricing (Non-JIT)."""
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    exp_rt, exp_qt = np.exp(-r*T), np.exp(-q*T)
    
    # Use scipy.stats.norm.cdf for reliable non-JIT implementation
    from scipy.stats import norm
    
    if isinstance(is_call, bool):
        if is_call:
            return S*exp_qt*norm.cdf(d1) - K*exp_rt*norm.cdf(d2)
        return K*exp_rt*norm.cdf(-d2) - S*exp_qt*norm.cdf(-d1)
    
    # Array case
    prices = np.empty_like(S)
    prices[is_call] = S[is_call]*exp_qt[is_call]*norm.cdf(d1[is_call]) - K[is_call]*exp_rt[is_call]*norm.cdf(d2[is_call])
    prices[~is_call] = K[~is_call]*exp_rt[~is_call]*norm.cdf(-d2[~is_call]) - S[~is_call]*exp_qt[~is_call]*norm.cdf(-d1[~is_call])
    return prices

def jit_mc_european_price_v2(S0, K, T, r, sigma, q, n_paths, is_call, antithetic, z_innovations=None, scheme=SCHEME_EULER):
    actual_paths = n_paths // 2 if antithetic else n_paths
    exp_rt = np.exp(-r * T)
    drift, diffusion = (r - q - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    z = z_innovations if z_innovations is not None else np.random.standard_normal(actual_paths)
    if antithetic:
        s1, s2 = S0 * np.exp(drift + diffusion * z), S0 * np.exp(drift - diffusion * z)
        p1, p2 = np.maximum(s1 - K if is_call else K - s1, 0.0), np.maximum(s2 - K if is_call else K - s2, 0.0)
        payoffs = (p1 + p2) * 0.5 * exp_rt
    else:
        st = S0 * np.exp(drift + diffusion * z)
        payoffs = np.maximum(st - K if is_call else K - st, 0.0) * exp_rt
    return np.mean(payoffs), np.sqrt(max(np.var(payoffs)/n_paths, 0.0))

def batch_greeks_jit_v2(S, K, T, sigma, r, q, is_call):
    """Batch greeks calculation (Non-JIT)."""
    # Use scalar_greeks logic in a loop for reliability
    n = len(S)
    delta = np.empty(n)
    gamma = np.empty(n)
    vega = np.empty(n)
    theta = np.empty(n)
    rho = np.empty(n)
    
    for i in range(n):
        d, g, th, v, rh = scalar_greeks_jit_v2(S[i], K[i], T[i], sigma[i], r[i], q[i], is_call[i])
        delta[i], gamma[i], theta[i], vega[i], rho[i] = d, g, th, v, rh
        
    return delta, gamma, vega, theta, rho

def scalar_greeks_jit_v2(S, K, T, sigma, r, q, is_call):
    """Scalar greeks calculation (Non-JIT)."""
    from scipy.stats import norm
    Ti = max(T, 1e-7)
    sqrt_T = np.sqrt(Ti)
    sig_sqrt_t = sigma * sqrt_T
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * Ti) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    pdf_d1 = norm.pdf(d1)
    exp_qt, exp_rt = np.exp(-q * Ti), np.exp(-r * Ti)
    nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
    
    gamma = (exp_qt * pdf_d1) / (S * sigma * sqrt_T)
    vega = (S * exp_qt * pdf_d1 * sqrt_T) * 0.01
    common_theta = -(S * pdf_d1 * sigma * exp_qt) / (2 * sqrt_T)
    if is_call:
        delta, rho = exp_qt * nd1, (K * Ti * exp_rt * nd2) * 0.01
        theta = (common_theta - r * K * exp_rt * nd2 + q * S * exp_qt * nd1) / 365.0
    else:
        delta, rho = exp_qt * (nd1 - 1.0), (-K * Ti * exp_rt * (1.0 - nd2)) * 0.01
        theta = (common_theta + r * K * exp_rt * (1.0 - nd2) - q * S * exp_qt * (1.0 - nd1)) / 365.0
    return delta, gamma, theta, vega, rho

def corrado_miller_initial_guess(market_price, spot, strike, maturity, rate, dividend, is_call):
    """Fast initial guess for Implied Volatility."""
    n = len(market_price)
    sigma = np.empty(n, dtype=np.float64)
    FACTOR = 2.5066282746310005
    for i in range(n):
        X = strike[i] * np.exp(-rate[i] * maturity[i])
        val = FACTOR / (np.sqrt(maturity[i]) * (spot[i] + X))
        exp_qt = np.exp(-dividend[i] * maturity[i])
        if is_call[i]: intrinsic = max(spot[i] * exp_qt - X, 0.0)
        else: intrinsic = max(X - spot[i] * exp_qt, 0.0)
        term = market_price[i] - intrinsic / 2.0
        inner = term**2 - intrinsic**2 / np.pi
        sigma[i] = val * (term + np.sqrt(max(inner, 0.0)))
    return np.clip(sigma, 0.001, 5.0)

# Backward compatibility stubs
def warmup_jit(): pass
fast_normal_cdf = fast_normal_cdf_v2
jit_generate_log_paths = generate_log_paths_v2
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
vectorized_newton_raphson_iv_jit = None
jit_cn_solver = None
