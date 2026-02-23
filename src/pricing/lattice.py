"""
Lattice Models for Option Pricing

Provides Binomial (CRR) and Trinomial tree models for pricing
European and American options. Optimized with NumPy for performance.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit

from .base import PricingStrategy
from .black_scholes import BlackScholesEngine
from .models import BSParameters


@dataclass
class LatticeGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class LatticeParameters(BSParameters):
    n_steps: int = 100




# =============================================================================
# OPTIMIZED: High-Performance Lattice Kernels (Numba JIT)
# =============================================================================


@njit(cache=True, fastmath=True)
def _binomial_jit_kernel(S0, K, T, r, q, sigma, n_steps, is_call, is_american):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    a = np.exp((r - q) * dt)
    p = (a - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal payoffs
    V = np.empty(n_steps + 1, dtype=np.float64)
    for j in range(n_steps + 1):
        st = S0 * (u ** (n_steps - j)) * (d**j)
        if is_call:
            V[j] = max(st - K, 0.0)
        else:
            V[j] = max(K - st, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            V_new = disc * (p * V[j] + (1 - p) * V[j + 1])
            if is_american:
                st = S0 * (u ** (i - j)) * (d**j)
                exercise = max(st - K, 0.0) if is_call else max(K - st, 0.0)
                V[j] = max(V_new, exercise)
            else:
                V[j] = V_new

    return V[0]


@njit(cache=True, fastmath=True)
def _trinomial_jit_kernel(S0, K, T, r, q, sigma, n_steps, is_call, is_american):
    dt = T / n_steps
    dx = sigma * np.sqrt(3 * dt)
    v_drift = r - q - 0.5 * sigma**2

    p_u = 0.5 * ((sigma**2 * dt + v_drift**2 * dt**2) / dx**2 + v_drift * dt / dx)
    p_d = 0.5 * ((sigma**2 * dt + v_drift**2 * dt**2) / dx**2 - v_drift * dt / dx)
    p_m = 1.0 - p_u - p_d
    disc = np.exp(-r * dt)

    num_nodes = 2 * n_steps + 1
    V = np.empty(num_nodes, dtype=np.float64)
    for j in range(num_nodes):
        st = S0 * np.exp(dx * (n_steps - j))
        V[j] = max(st - K, 0.0) if is_call else max(K - st, 0.0)

    for i in range(n_steps - 1, -1, -1):
        for j in range(2 * i + 1):
            V_new = disc * (p_u * V[j] + p_m * V[j + 1] + p_d * V[j + 2])
            if is_american:
                st = S0 * np.exp(dx * (i - j))
                exercise = max(st - K, 0.0) if is_call else max(K - st, 0.0)
                V[j] = max(V_new, exercise)
            else:
                V[j] = V_new

    return V[0]


def validate_convergence(
    spot, strike, maturity, volatility, rate, dividend, option_type, step_sizes
):
    """Validate that the pricing method is converging as steps increase."""
    bs_params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
    engine = BlackScholesEngine()
    if option_type == "call":
        bs_price = float(engine.price(params=bs_params, option_type="call"))
    else:
        bs_price = float(engine.price(params=bs_params, option_type="put"))

    bin_errors = []
    tri_errors = []

    for n_s in step_sizes:
        bin_pricer = BinomialTreePricer(n_steps=n_s)
        tri_pricer = TrinomialTreePricer(n_steps=n_s)

        bin_errors.append(abs(bin_pricer.price(bs_params, option_type) - bs_price))
        tri_errors.append(abs(tri_pricer.price(bs_params, option_type) - bs_price))

    return {"binomial_errors": bin_errors, "trinomial_errors": tri_errors}


class BinomialTreePricer(PricingStrategy):
    """Cox-Ross-Rubinstein (CRR) JIT Pricer."""

    def __init__(
        self,
        n_steps: int = 100,
        exercise_type: Literal["european", "american"] = "european",
    ):
        self.n_steps = n_steps
        self.exercise_type = exercise_type.lower()

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        return float(
            _binomial_jit_kernel(
                params.spot,
                params.strike,
                params.maturity,
                params.rate,
                params.dividend,
                params.volatility,
                self.n_steps,
                option_type.lower() == "call",
                self.exercise_type == "american",
            )
        )


class TrinomialTreePricer(PricingStrategy):
    """Standard Trinomial Tree JIT Pricer."""

    def __init__(
        self,
        n_steps: int = 100,
        exercise_type: Literal["european", "american"] = "european",
    ):
        self.n_steps = n_steps
        self.exercise_type = exercise_type.lower()

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        return float(
            _trinomial_jit_kernel(
                params.spot,
                params.strike,
                params.maturity,
                params.rate,
                params.dividend,
                params.volatility,
                self.n_steps,
                option_type.lower() == "call",
                self.exercise_type == "american",
            )
        )
