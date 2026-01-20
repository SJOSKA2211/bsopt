"""
Crank-Nicolson Finite Difference Solver for Black-Scholes PDE

Provides numerical pricing for European options by solving the Black-Scholes
partial differential equation using the second-order accurate Crank-Nicolson scheme.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg, spilu, spsolve

from .base import PricingStrategy
from src.pricing.models import BSParameters, OptionGreeks

logger = logging.getLogger(__name__)


class CrankNicolsonSolver(PricingStrategy):
    """
    Finite difference solver using the Crank-Nicolson method.
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
        self.use_iterative = use_iterative

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

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Implementation of PricingStrategy interface."""
        self._setup_grid(params)
        self.option_type = option_type.lower()
        return self.get_greeks()

    def _build_matrices(self) -> Tuple[Any, Any]:
        """
        Build the implicit (A) and explicit (B) matrices for Crank-Nicolson.
        """
        M = self.n_spots
        dt = self.dt
        dS = self.dS

        # Coefficients for the tridiagonal system
        # α_i, β_i, γ_i as defined in the documentation
        indices = np.arange(1, M)
        S_i = self.s_grid[indices]

        sig2 = self.volatility**2
        mu = self.rate - self.dividend

        alpha = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) - mu * S_i / dS)
        beta = -0.5 * dt * (sig2 * (S_i**2) / (dS**2) + self.rate)
        gamma = 0.25 * dt * (sig2 * (S_i**2) / (dS**2) + mu * S_i / dS)

        # Implicit matrix A (tridiag(-α, 1-β, -γ))
        # Note: we use indices 1 to M-1 for internal points
        diag_A = 1.0 - beta
        lower_A = -alpha[1:]
        upper_A = -gamma[:-1]

        A = diags([lower_A, diag_A, upper_A], [-1, 0, 1], shape=(M - 1, M - 1), format="csr")

        # Explicit matrix B (tridiag(α, 1+β, γ))
        diag_B = 1.0 + beta
        lower_B = alpha[1:]
        upper_B = gamma[:-1]

        B = diags([lower_B, diag_B, upper_B], [-1, 0, 1], shape=(M - 1, M - 1), format="csr")

        return A, B

    def _solve_pde(self) -> np.ndarray:
        """Internal core solver for the Black-Scholes PDE."""
        M = self.n_spots
        N = self.n_time
        dt = self.dt

        # 1. Initialize terminal condition (payoff at t=T)
        if self.option_type == "call":
            V = np.maximum(self.s_grid - self.strike, 0.0)
        else:
            V = np.maximum(self.strike - self.s_grid, 0.0)

        # 2. Build matrices
        A, B = self._build_matrices()

        # Preconditioner for iterative solver
        M_op = None
        current_use_iterative = self.use_iterative
        if current_use_iterative:
            try:
                # Incomplete LU factorization for preconditioning
                ilu = spilu(A, drop_tol=1e-4)
                M_op = LinearOperator(A.shape, ilu.solve)
            except RuntimeError as e:
                logger.warning(f"Sparse ILU failed, falling back to direct solver: {e}")
                current_use_iterative = False

        # 3. Time stepping (backward from T to 0)
        for n in range(N, 0, -1):
            tau = (N - n + 1) * dt

            # Boundary conditions at level n-1
            if self.option_type == "call":
                v_min_next = 0.0
                v_max_next = self.s_max - self.strike * np.exp(-self.rate * tau)
            else:
                v_min_next = self.strike * np.exp(-self.rate * tau)
                v_max_next = 0.0

            # Current boundary conditions
            if self.option_type == "call":
                v_min_curr = 0.0
                v_max_curr = self.s_max - self.strike * np.exp(-self.rate * (tau - dt))
            else:
                v_min_curr = self.strike * np.exp(-self.rate * (tau - dt))
                v_max_curr = 0.0

            # RHS: b = B * V_internal + boundary_contributions
            b = B @ V[1:M]

            # alpha[0] corresponds to i=1
            S_1 = self.s_grid[1]
            S_M_1 = self.s_grid[M - 1]
            sig2 = self.volatility**2
            mu = self.rate - self.dividend

            alpha_1 = 0.25 * dt * (sig2 * (S_1**2) / (self.dS**2) - mu * S_1 / self.dS)
            gamma_M_1 = 0.25 * dt * (sig2 * (S_M_1**2) / (self.dS**2) + mu * S_M_1 / self.dS)

            b[0] += alpha_1 * (v_min_curr + v_min_next)
            b[-1] += gamma_M_1 * (v_max_curr + v_max_next)

            # Solve A * V_new = b
            if current_use_iterative:
                # Use both for compatibility across scipy versions if possible, or just rtol
                V_internal, info = cg(A, b, x0=V[1:M], M=M_op, rtol=1e-6)
                if info != 0:
                    logger.warning(f"Iterative solver failed to converge at step {n} (info={info})")
                V[1:M] = V_internal
            else:
                V[1:M] = spsolve(A, b)

            V[0] = v_min_next
            V[M] = v_max_next

        return V

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
            delta = 1.0 if (self.option_type == "call" and self.spot > self.strike) else 0.0
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
            (self.price(params_up, self.option_type) - self.price(params_down, self.option_type))
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
                    self.spot, self.strike, self.maturity, self.volatility, self.rate, self.dividend
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

    def get_diagnostics(self) -> Dict[str, Any]:
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
            "accuracy": {"spatial_order": 2, "temporal_order": 2, "note": "O(dt^2 + dS^2)"},
            "performance": {
                "operations_per_step": "O(M)",
                "total_operations": f"O({self.n_spots * self.n_time})",
                "sparse_matrix": True,
            },
        }
