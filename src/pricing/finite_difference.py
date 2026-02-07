"""
Crank-Nicolson Finite Difference Solver for Black-Scholes PDE

Provides numerical pricing for European options by solving the Black-Scholes
partial differential equation using the second-order accurate Crank-Nicolson scheme.
"""

import logging
from typing import Any

import numpy as np

from src.pricing.models import BSParameters, OptionGreeks
from src.pricing.quant_utils import jit_cn_solver

from .base import PricingStrategy

logger = logging.getLogger(__name__)


class CrankNicolsonSolver(PricingStrategy):
    """
    Finite difference solver using the Crank-Nicolson method.
    Optimized using Numba JIT compilation.
    """

    def __init__(
        self,
        n_spots: int = 200,
        n_time: int = 500,
        s_max_mult: float = 3.0,
        use_iterative: bool = False,
    ):
        self.n_spots = n_spots
        self.n_time = n_time
        self.s_max_mult = s_max_mult
        self.use_iterative = use_iterative  # Kept for API compatibility, but JIT uses Thomas algo (direct)

    def _setup_grid(self, params: BSParameters):
        """Initialize grid for a specific option."""
        self.spot = float(params.spot)
        self.strike = float(params.strike)
        self.maturity = float(params.maturity)
        self.volatility = float(params.volatility)
        self.rate = float(params.rate)
        self.dividend = float(params.dividend)

        # Domain boundaries
        self.s_min = 0.0
        self.s_max = self.strike * self.s_max_mult

        # Grid spacing
        self.dS = self.s_max / self.n_spots
        self.dt = self.maturity / self.n_time if self.maturity > 0 else 0.0

        # Grids
        self.s_grid = np.linspace(self.s_min, self.s_max, self.n_spots + 1)

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Implementation of PricingStrategy interface."""
        self._setup_grid(params)
        self.option_type = option_type.lower()

        if self.maturity <= 1e-12:
            if self.option_type == "call":
                return max(self.spot - self.strike, 0.0)
            else:
                return max(self.strike - self.spot, 0.0)

        V = self._solve_pde()
        return float(np.interp(self.spot, self.s_grid, V))

    def calculate_greeks(
        self, params: BSParameters, option_type: str = "call"
    ) -> OptionGreeks:
        """Implementation of PricingStrategy interface."""
        self._setup_grid(params)
        self.option_type = option_type.lower()
        return self.get_greeks()

    def _solve_pde(self) -> np.ndarray:
        """Internal core solver for the Black-Scholes PDE using JIT."""
        is_call = self.option_type == "call"

        # Delegate to Numba-optimized solver
        return jit_cn_solver(
            self.s_grid,
            self.strike,
            self.maturity,
            self.rate,
            self.volatility,
            self.dividend,
            is_call,
            self.n_time,
        )

    def solve(self) -> float:
        """
        Solve the PDE and return the option price at the current spot.
        """
        if self.maturity <= 1e-12:
            if self.option_type == "call":
                return max(self.spot - self.strike, 0.0)
            else:
                return max(self.strike - self.spot, 0.0)

        V = self._solve_pde()
        return float(np.interp(self.spot, self.s_grid, V))

    def get_greeks(self) -> OptionGreeks:
        """
        Calculate Greeks using finite differences on the solution.
        """
        # Handle zero maturity case
        if self.maturity <= 1e-12:
            delta = (
                1.0 if (self.option_type == "call" and self.spot > self.strike) else 0.0
            )
            if self.option_type == "put" and self.spot < self.strike:
                delta = -1.0
            return OptionGreeks(delta, 0.0, 0.0, 0.0, 0.0)

        # 1. Delta and Gamma from current solve
        V = self._solve_pde()

        # Find index closest to spot
        idx = int(np.searchsorted(self.s_grid, self.spot))
        if idx == 0:
            idx = 1
        if idx >= self.n_spots:
            idx = self.n_spots - 1

        # Use central differences on grid if possible, otherwise interpolation
        delta = (V[idx + 1] - V[idx - 1]) / (2 * self.dS)
        gamma = (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / (self.dS**2)

        # 2. Vega (bump vol by 1%)
        eps_vol = 0.01

        params_up = BSParameters(
            self.spot,
            self.strike,
            self.maturity,
            self.volatility + eps_vol,
            self.rate,
            self.dividend,
        )
        params_down = BSParameters(
            self.spot,
            self.strike,
            self.maturity,
            self.volatility - eps_vol,
            self.rate,
            self.dividend,
        )

        vega = (
            (
                self.price(params_up, self.option_type)
                - self.price(params_down, self.option_type)
            )
            / (2 * eps_vol)
            * 0.01
        )

        # 3. Theta (1 day decay)
        dt_day = 1.0 / 365.0
        if self.maturity > dt_day:
            params_t = BSParameters(
                self.spot,
                self.strike,
                self.maturity - dt_day,
                self.volatility,
                self.rate,
                self.dividend,
            )
            theta = self.price(params_t, self.option_type) - self.price(
                BSParameters(
                    self.spot,
                    self.strike,
                    self.maturity,
                    self.volatility,
                    self.rate,
                    self.dividend,
                ),
                self.option_type,
            )
        else:
            theta = 0.0

        # 4. Rho (bump rate by 1%)
        eps_rate = 0.01
        params_r_up = BSParameters(
            self.spot,
            self.strike,
            self.maturity,
            self.volatility,
            self.rate + eps_rate,
            self.dividend,
        )
        params_r_down = BSParameters(
            self.spot,
            self.strike,
            self.maturity,
            self.volatility,
            self.rate - eps_rate,
            self.dividend,
        )

        rho = (
            (
                self.price(params_r_up, self.option_type)
                - self.price(params_r_down, self.option_type)
            )
            / (2 * eps_rate)
            * 0.01
        )

        return OptionGreeks(delta, gamma, theta, vega, rho)

    def _clone(self, **kwargs):
        """Clone the solver with some modified grid parameters."""
        # Note: This now only clones grid configuration, as option params are passed to price()
        params = {
            "n_spots": self.n_spots,
            "n_time": self.n_time,
            "s_max_mult": self.s_max_mult,
            "use_iterative": self.use_iterative,
        }
        params.update(kwargs)
        return CrankNicolsonSolver(**params)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return solver diagnostics."""
        return {
            "scheme": "Crank-Nicolson",
            "grid_spacing": {
                "n_spots": self.n_spots,
                "n_time": self.n_time,
                "dS": self.dS,
                "dt": self.dt,
            },
            "domain": {"S_min": self.s_min, "S_max": self.s_max, "T": self.maturity},
            "stability": {
                "mesh_ratio": self.dt / (self.dS**2) if self.dS > 0 else 0,
                "max_explicit_dt": (
                    1.0 / (2.0 * self.volatility**2 * (self.s_max / self.dS) ** 2)
                    if self.volatility > 0 and self.dS > 0
                    else 0
                ),
                "is_stable": True,
                "note": "Unconditionally stable",
            },
            "accuracy": {
                "spatial_order": 2,
                "temporal_order": 2,
                "note": "O(dt^2 + dS^2)",
            },
            "performance": {
                "operations_per_step": "O(M)",
                "total_operations": f"O({self.n_spots * self.n_time})",
                "sparse_matrix": True,
            },
        }
