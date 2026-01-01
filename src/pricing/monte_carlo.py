"""
Enhanced Monte Carlo Option Pricing Engine

Provides European and American option pricing using:
- Quasi-Monte Carlo (Sobol sequences)
- Variance reduction (Antithetic and Control Variates)
- Numba JIT acceleration
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, cast

import numpy as np
from numba import njit, prange
from scipy.stats import norm, qmc

from src.pricing.base import PricingStrategy
from src.pricing.black_scholes import BSParameters, OptionGreeks


@dataclass
class MCConfig:
    """Configuration for Monte Carlo simulation."""

    n_paths: int = 100000
    n_steps: int = 252
    method: Literal["pseudo", "sobol"] = "pseudo"
    antithetic: bool = True
    control_variate: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

        if self.method == "sobol":
            self.n_paths = 2 ** int(np.ceil(np.log2(self.n_paths)))

        if self.antithetic and self.n_paths % 2 != 0:
            self.n_paths += 1


@njit(parallel=True, cache=True)
def _simulate_paths_jit(
    spot, maturity, volatility, rate, dividend, n_paths, n_steps, dt, random_normals
):
    """Simulate asset paths using exact GBM discretization."""
    paths = np.empty((n_paths, n_steps + 1))
    drift = (rate - dividend - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)
    for i in prange(n_paths):
        paths[i, 0] = spot
        for j in range(n_steps):
            paths[i, j + 1] = paths[i, j] * np.exp(drift + diffusion * random_normals[i, j])
    return paths


@njit(cache=True)
def _laguerre_basis(x, degree=3):
    """Compute polynomial basis for LSM regression {1, x, x^2-1, x^3-3x}."""
    basis = np.empty((len(x), degree + 1))
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = x
    if degree >= 2:
        basis[:, 2] = x**2 - 1.0
    if degree >= 3:
        basis[:, 3] = x**3 - 3.0 * x

    return basis


@njit(cache=True)
def _lsm_regression_jit(paths, payoffs, discount_factor, n_steps, n_paths):
    """JIT-accelerated Longstaff-Schwartz regression core."""
    cash_flows = np.copy(payoffs[:, -1])
    for t in range(n_steps - 1, 0, -1):
        itm = payoffs[:, t] > 0
        if np.sum(itm) < 10:
            cash_flows *= discount_factor
            continue
        x = paths[itm, t]
        y = cash_flows[itm] * discount_factor

        # Polynomial fit (simple order 3)
        A = np.vander(x, 4)
        beta = np.linalg.solve(A.T @ A, A.T @ y)
        continuation_value = A @ beta

        exercise = payoffs[itm, t]
        should_exercise = exercise > continuation_value

        # Correctly update cash flows for those who exercise
        itm_indices = np.where(itm)[0]
        for i in range(len(itm_indices)):
            idx = itm_indices[i]
            if should_exercise[i]:
                cash_flows[idx] = exercise[i]
            else:
                cash_flows[idx] *= discount_factor

        # For those not ITM, just discount
        not_itm_indices = np.where(~itm)[0]
        for idx in not_itm_indices:
            cash_flows[idx] *= discount_factor

    return np.mean(cash_flows * discount_factor)


def geometric_asian_price(params: BSParameters, option_type: str, n_steps: int) -> float:
    """Analytical price for Geometric Asian option (Control Variate)."""
    T, sigma, r, q, S, K = (
        params.maturity,
        params.volatility,
        params.rate,
        params.dividend,
        params.spot,
        params.strike,
    )
    sigma_a = sigma * np.sqrt((2 * n_steps + 1) / (6 * (n_steps + 1)))
    mu_a = (r - q - 0.5 * sigma**2) * 0.5 + 0.5 * sigma_a**2
    d1 = (np.log(S / K) + (mu_a + 0.5 * sigma_a**2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)
    if option_type.lower() == "call":
        return float(
            np.exp(-r * T)
            * (S * np.exp(mu_a * T + 0.5 * sigma_a**2 * T) * norm.cdf(d1) - K * norm.cdf(d2))
        )
    return float(
        np.exp(-r * T)
        * (K * norm.cdf(-d2) - S * np.exp(mu_a * T + 0.5 * sigma_a**2 * T) * norm.cdf(-d1))
    )


class MonteCarloEngine(PricingStrategy):
    def __init__(self, config: Optional[MCConfig] = None):
        self.config = config or MCConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Implementation of PricingStrategy interface."""
        price, _ = self.price_european(params, option_type)
        return price

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Implementation of PricingStrategy interface using finite differences."""
        # Simple finite difference implementation for Greeks in MC
        eps = max(params.spot * 0.01, 0.01)

        # Delta
        p_plus = self.price(
            BSParameters(
                params.spot + eps,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            ),
            option_type,
        )
        p_minus = self.price(
            BSParameters(
                params.spot - eps,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            ),
            option_type,
        )
        delta = (p_plus - p_minus) / (2 * eps)

        # Gamma
        p_mid = self.price(params, option_type)
        gamma = (p_plus - 2 * p_mid + p_minus) / (eps**2)

        # Vega
        eps_v = 0.01
        p_v_plus = self.price(
            BSParameters(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility + eps_v,
                params.rate,
                params.dividend,
            ),
            option_type,
        )
        p_v_minus = self.price(
            BSParameters(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility - eps_v,
                params.rate,
                params.dividend,
            ),
            option_type,
        )
        vega = (p_v_plus - p_v_minus) / (2 * eps_v) * 0.01

        # Theta
        dt = 1.0 / 365.0
        if params.maturity > dt:
            p_t = self.price(
                BSParameters(
                    params.spot,
                    params.strike,
                    params.maturity - dt,
                    params.volatility,
                    params.rate,
                    params.dividend,
                ),
                option_type,
            )
            theta = p_t - p_mid
        else:
            theta = 0.0

        # Rho
        eps_r = 0.0001
        p_r_plus = self.price(
            BSParameters(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate + eps_r,
                params.dividend,
            ),
            option_type,
        )
        p_r_minus = self.price(
            BSParameters(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate - eps_r,
                params.dividend,
            ),
            option_type,
        )
        rho = (p_r_plus - p_r_minus) / (2 * eps_r) * 0.01

        return OptionGreeks(delta, gamma, vega, theta, rho)

    def _generate_random_normals(self, n_paths: int, n_steps: int) -> np.ndarray:
        if self.config.method == "sobol":
            sampler = qmc.Sobol(d=n_steps, scramble=True, seed=self.config.seed)
            if self.config.antithetic:
                u = sampler.random(n_paths // 2)
                z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
                return np.vstack((z, -z))
            return cast(np.ndarray, norm.ppf(np.clip(sampler.random(n_paths), 1e-10, 1 - 1e-10)))
        z = (
            self.rng.standard_normal((n_paths // 2, n_steps))
            if self.config.antithetic
            else self.rng.standard_normal((n_paths, n_steps))
        )
        return np.vstack((z, -z)) if self.config.antithetic else z

    def price_european(
        self, params: BSParameters, option_type: str = "call"
    ) -> Tuple[float, float]:
        option_type = option_type.lower()
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        dt = params.maturity / self.config.n_steps
        z = self._generate_random_normals(self.config.n_paths, self.config.n_steps)
        paths = _simulate_paths_jit(
            params.spot,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            self.config.n_paths,
            self.config.n_steps,
            dt,
            z,
        )

        payoffs = (
            np.maximum(paths[:, -1] - params.strike, 0.0)
            if option_type == "call"
            else np.maximum(params.strike - paths[:, -1], 0.0)
        )
        discounted = payoffs * np.exp(-params.rate * params.maturity)

        if self.config.control_variate:
            # Use terminal price S_T as control variate
            # E[S_T * e^-rT] = S_0 * e^-qT
            X = paths[:, -1] * np.exp(-params.rate * params.maturity)
            expected_X = params.spot * np.exp(-params.dividend * params.maturity)

            cov_matrix = np.cov(discounted, X)
            c = -cov_matrix[0, 1] / cov_matrix[1, 1] if abs(cov_matrix[1, 1]) > 1e-10 else 0.0
            discounted = discounted + c * (X - expected_X)

        return float(np.mean(discounted)), float(
            1.96 * np.std(discounted) / np.sqrt(self.config.n_paths)
        )

    def price_american_lsm(self, params: BSParameters, option_type: str = "put") -> float:
        option_type = option_type.lower()
        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        dt = params.maturity / self.config.n_steps
        z = self._generate_random_normals(self.config.n_paths, self.config.n_steps)
        paths = _simulate_paths_jit(
            params.spot,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            self.config.n_paths,
            self.config.n_steps,
            dt,
            z,
        )
        payoffs = (
            np.maximum(paths - params.strike, 0.0)
            if option_type.lower() == "call"
            else np.maximum(params.strike - paths, 0.0)
        )
        return float(
            _lsm_regression_jit(
                paths, payoffs, np.exp(-params.rate * dt), self.config.n_steps, self.config.n_paths
            )
        )
