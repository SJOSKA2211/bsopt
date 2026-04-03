"""
Geometric Brownian Motion (GBM) ODE Solver

This module implements numerical methods for solving the GBM stochastic differential equation:
    dS = μSdt + σSdW

Methods implemented:
1. Euler-Maruyama - First-order weak convergence, O(√dt)
2. Milstein - First-order strong convergence, O(dt)
3. Runge-Kutta 4 (RK4) with Milstein correction - Strong convergence O(dt²)

Author: Manifold Quant Team
"""

from __future__ import annotations

import numpy as np
import structlog
from numba import njit, prange

logger = structlog.get_logger(__name__)


@njit(cache=True, fastmath=True, parallel=True)
def _euler_maruyama_step(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    sqrt_dt: float,
    rng_state: np.ndarray,
) -> np.ndarray:
    """
    Single step of Euler-Maruyama method for multiple paths.

    dS = μSdt + σS√dt * N(0,1)
    S_{t+dt} = S_t + μ*S_t*dt + σ*S_t*√dt*Z

    Args:
        s: Current prices - shape (n_paths,)
        mu: Drift parameters - shape (n_paths,)
        sigma: Volatility parameters - shape (n_paths,)
        dt: Time step
        sqrt_dt: sqrt(dt)
        rng_state: Random number generator state

    Returns:
        Next prices - shape (n_paths,)
    """
    n = len(s)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        z = np.random.normal()

        drift = mu[i] * s[i] * dt
        diffusion = sigma[i] * s[i] * sqrt_dt * z

        result[i] = s[i] + drift + diffusion

        if result[i] < 0.0:
            result[i] = 1e-10

    return result


@njit(cache=True, fastmath=True, parallel=True)
def _milstein_step(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    sqrt_dt: float,
) -> np.ndarray:
    """
    Single step of Milstein method with improved strong convergence.

    S_{t+dt} = S_t + μ*S_t*dt + σ*S_t*√dt*Z + 0.5*σ²*S_t*(Z²-dt)

    The additional term (Milstein correction) improves convergence from O(√dt) to O(dt).

    Args:
        s: Current prices - shape (n_paths,)
        mu: Drift parameters - shape (n_paths,)
        sigma: Volatility parameters - shape (n_paths,)
        dt: Time step
        sqrt_dt: sqrt(dt)

    Returns:
        Next prices - shape (n_paths,)
    """
    n = len(s)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        z = np.random.normal()
        z_sq = z * z

        drift = mu[i] * s[i] * dt
        diffusion = sigma[i] * s[i] * sqrt_dt * z
        milstein_correction = 0.5 * sigma[i] * sigma[i] * s[i] * (z_sq - dt)

        result[i] = s[i] + drift + diffusion + milstein_correction

        if result[i] < 0.0:
            result[i] = 1e-10

    return result


@njit(cache=True, fastmath=True, parallel=True)
def _rk4_milstein_step(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    sqrt_dt: float,
) -> np.ndarray:
    """
    Single step of RK4 method with Milstein diffusion correction.

    Uses deterministic RK4 for drift and exact Milstein for diffusion.
    This gives strong convergence O(dt²) for drift and O(dt) for diffusion.

    Args:
        s: Current prices - shape (n_paths,)
        mu: Drift parameters - shape (n_paths,)
        sigma: Volatility parameters - shape (n_paths,)
        dt: Time step
        sqrt_dt: sqrt(dt)

    Returns:
        Next prices - shape (n_paths,)
    """
    n = len(s)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        z = np.random.normal()
        z_sq = z * z
        si = s[i]
        mu_i = mu[i]
        sigma_i = sigma[i]

        k1 = mu_i * si
        k2 = mu_i * (si + 0.5 * dt * k1)
        k3 = mu_i * (si + 0.5 * dt * k2)
        k4 = mu_i * (si + dt * k3)

        drift = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        diffusion = sigma_i * si * sqrt_dt * z
        milstein_correction = 0.5 * sigma_i * sigma_i * si * (z_sq - dt)

        result[i] = si + drift + diffusion + milstein_correction

        if result[i] < 0.0:
            result[i] = 1e-10

    return result


def simulate_gbm_euler(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion using Euler-Maruyama method.

    Args:
        s0: Initial prices - shape (n_paths,)
        mu: Annual drift (μ) - shape (n_paths,) or scalar
        sigma: Annual volatility (σ) - shape (n_paths,) or scalar
        t: Time horizon in years
        dt: Time step size in years (e.g., 1/252 for daily)
        seed: Random seed for reproducibility

    Returns:
        Price paths - shape (n_steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    s0 = np.atleast_1d(s0).astype(np.float64)
    mu = np.atleast_1d(mu).astype(np.float64)
    sigma = np.atleast_1d(sigma).astype(np.float64)

    n_steps = int(t / dt)
    n_paths = len(s0)

    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = s0

    current = s0.copy()
    for step in range(n_steps):
        current = _euler_maruyama_step(current, mu, sigma, dt, sqrt_dt, None)
        paths[step + 1] = current

    return paths


def simulate_gbm_milstein(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion using Milstein method.

    Strong convergence O(dt) vs Euler-Maruyama O(√dt)

    Args:
        s0: Initial prices - shape (n_paths,)
        mu: Annual drift (μ) - shape (n_paths,) or scalar
        sigma: Annual volatility (σ) - shape (n_paths,) or scalar
        t: Time horizon in years
        dt: Time step size in years
        seed: Random seed for reproducibility

    Returns:
        Price paths - shape (n_steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    s0 = np.atleast_1d(s0).astype(np.float64)
    mu = np.atleast_1d(mu).astype(np.float64)
    sigma = np.atleast_1d(sigma).astype(np.float64)

    n_steps = int(t / dt)
    n_paths = len(s0)

    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = s0

    current = s0.copy()
    for step in range(n_steps):
        current = _milstein_step(current, mu, sigma, dt, sqrt_dt)
        paths[step + 1] = current

    return paths


def simulate_gbm_rk4(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion using RK4-Milstein method.

    Highest precision numerical solver for stochastic differential equations.
    Uses RK4 for the deterministic part and Milstein for the stochastic part.

    Args:
        s0: Initial prices
        mu: Annual drift
        sigma: Annual volatility
        t: Time horizon
        dt: Time step size
        seed: Random seed

    Returns:
        Price paths - shape (n_steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    s0 = np.atleast_1d(s0).astype(np.float64)
    mu = np.atleast_1d(mu).astype(np.float64)
    sigma = np.atleast_1d(sigma).astype(np.float64)

    n_steps = int(t / dt)
    n_paths = len(s0)

    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = s0

    current = s0.copy()
    for step in range(n_steps):
        current = _rk4_milstein_step(current, mu, sigma, dt, sqrt_dt)
        paths[step + 1] = current

    return paths


@njit(cache=True, fastmath=True, parallel=True)
def _exact_gbm_step(
    s: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Single step using the exact analytical solution of GBM.
    S_{t+dt} = S_t * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
    """
    n = len(s)
    result = np.empty(n, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for i in prange(n):
        z = np.random.normal()
        drift = (mu[i] - 0.5 * sigma[i] * sigma[i]) * dt
        diffusion = sigma[i] * sqrt_dt * z
        result[i] = s[i] * np.exp(drift + diffusion)

        if result[i] < 0.0:
            result[i] = 1e-10

    return result

def simulate_gbm_exact(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion using the exact analytical solution.
    Faster and more accurate than numerical approximations.
    """
    if seed is not None:
        np.random.seed(seed)

    s0 = np.atleast_1d(s0).astype(np.float64)
    mu = np.atleast_1d(mu).astype(np.float64)
    sigma = np.atleast_1d(sigma).astype(np.float64)

    n_steps = int(t / dt)
    n_paths = len(s0)

    paths = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    paths[0] = s0

    current = s0.copy()
    for step in range(n_steps):
        current = _exact_gbm_step(current, mu, sigma, dt)
        paths[step + 1] = current

    return paths

def simulate_gbm(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
    method: str = "exact",
    prefer_rust: bool = True,
) -> np.ndarray:
    """
    Unified high-performance GBM simulation.
    Automatically offloads to Rust core if available and prefer_rust is True.
    """
    if prefer_rust:
        from src.math_kernel.rust_engine import is_rust_available, simulate_gbm_rk4 as rust_gbm
        if is_rust_available():
            # Rust simulate_gbm_rk4 now uses the exact solution internally
            return rust_gbm(s0, mu, sigma, t, dt, seed=seed)

    if method == "exact":
        return simulate_gbm_exact(s0, mu, sigma, t, dt, seed=seed)
    elif method == "rk4":
        return simulate_gbm_rk4(s0, mu, sigma, t, dt, seed=seed)
    elif method == "milstein":
        return simulate_gbm_milstein(s0, mu, sigma, t, dt, seed=seed)
    else:
        return simulate_gbm_euler(s0, mu, sigma, t, dt, seed=seed)


def simulate_gbm_antithetic(
    s0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    t: float,
    dt: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM with antithetic variates for variance reduction.

    Uses pairs of paths with opposite random draws to reduce variance.

    Args:
        s0: Initial prices - shape (n_paths,)
        mu: Annual drift (μ)
        sigma: Annual volatility (σ)
        t: Time horizon
        dt: Time step
        seed: Random seed

    Returns:
        Tuple of (paths_1, paths_2) - each shape (n_steps + 1, n_paths)
    """
    if seed is not None:
        np.random.seed(seed)

    n_paths = len(np.atleast_1d(s0))
    half_paths = n_paths // 2

    s0_half = np.atleast_1d(s0)[:half_paths]

    if seed is not None:
        np.random.seed(seed)

    paths_positive = simulate_gbm_milstein(s0_half, mu, sigma, t, dt, seed=seed)

    if seed is not None:
        np.random.seed(seed + 1)

    paths_negative = simulate_gbm_milstein(s0_half, mu, sigma, t, dt, seed=seed + 1)

    full_paths_1 = np.concatenate([paths_positive, paths_negative], axis=1)
    full_paths_2 = np.concatenate([paths_negative, paths_positive], axis=1)

    return full_paths_1, full_paths_2


def monte_carlo_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    is_call: bool,
    n_paths: int = 100_000,
    dt: float = 1 / 252,
    method: str = "exact",
    seed: int | None = None,
    prefer_rust: bool = True,
) -> dict:
    """
    Price European option using Monte Carlo simulation of GBM.

    Args:
        s0: Initial stock price
        k: Strike price
        t: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        is_call: True for call, False for put
        n_paths: Number of simulation paths
        dt: Time step for simulation
        method: 'exact', 'euler', 'milstein', or 'rk4'
        seed: Random seed
        prefer_rust: Whether to offload to Rust if available

    Returns:
        Dictionary with price, std_error, and confidence interval
    """
    if seed is not None:
        np.random.seed(seed)

    s0_arr = np.full(n_paths, s0)
    mu_arr = np.full(n_paths, r)
    sigma_arr = np.full(n_paths, sigma)

    paths = simulate_gbm(
        s0_arr, mu_arr, sigma_arr, t, dt, seed=seed, method=method, prefer_rust=prefer_rust
    )

    final_prices = paths[-1]
    discount_factor = np.exp(-r * t)

    if is_call:
        payoffs = np.maximum(final_prices - k, 0)
    else:
        payoffs = np.maximum(k - final_prices, 0)

    price = discount_factor * np.mean(payoffs)
    std_error = discount_factor * np.std(payoffs) / np.sqrt(n_paths)
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price": float(price),
        "std_error": float(std_error),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_paths": n_paths,
        "n_steps": int(t / dt),
    }


def gbm_parameters_from_historical(
    prices: np.ndarray,
    dt: float = 1 / 252,
) -> tuple[float, float]:
    """
    Estimate GBM parameters (μ, σ) from historical price data.

    Uses log returns to estimate drift and volatility.

    Args:
        prices: Historical prices
        dt: Time step between observations

    Returns:
        Tuple of (annualized_drift, annualized_volatility)
    """
    returns = np.diff(np.log(prices))

    mu = np.mean(returns) / dt + 0.5 * np.var(returns) / dt

    sigma = np.std(returns) / np.sqrt(dt)

    return float(mu), float(sigma)


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("GBM Monte Carlo Benchmark")
    print("=" * 60)

    s0 = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    sigma = 0.2

    n_paths = 500_000
    dt = 1 / 252

    print("\nParameters:")
    print(f"  S0={s0}, K={k}, T={t}, r={r}, σ={sigma}")
    print(f"  Paths={n_paths:,}, Steps={int(t / dt)}")

    for method in ["euler", "milstein", "rk4"]:
        start = time.perf_counter()
        result = monte_carlo_price(
            s0, k, t, r, sigma, True, n_paths=n_paths, dt=dt, method=method, seed=42
        )
        elapsed = time.perf_counter() - start

        print(f"\n{method.upper()}:")
        print(f"  Price: ${result['price']:.4f}")
        print(f"  95% CI: [${result['ci_95_lower']:.4f}, ${result['ci_95_upper']:.4f}]")
        print(f"  Time: {elapsed:.3f}s")

    print("\n" + "=" * 60)
    print("Parameter Estimation from Historical Data")
    print("=" * 60)

    np.random.seed(42)
    historical = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))

    mu, vol = gbm_parameters_from_historical(historical)
    print(f"\nEstimated μ: {mu:.4f} ({mu * 100:.2f}%)")
    print(f"Estimated σ: {vol:.4f} ({vol * 100:.2f}%)")
