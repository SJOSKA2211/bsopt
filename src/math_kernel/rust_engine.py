"""
Rust-Accelerated Pricing Engine

Implements the high-performance CPU parallel path using equaflow_core (Rust/PyO3).
"""

import numpy as np
import structlog

from src.math_kernel.base import BasePricingEngine
from src.math_kernel.models import BSParameters

logger = structlog.get_logger(__name__)


class RustPricingEngine(BasePricingEngine):
    """
    CPU-bound parallel pricing engine powered by Rust.
    Optimized for massive batches where GPU transfer overhead outweighs compute.
    """

    def __init__(self):
        try:
            import equaflow_core

            self.core = equaflow_core
            self.available = True
        except ImportError:
            logger.warning("equaflow_core_not_found_rust_engine_disabled")
            self.available = False

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Scalar pricing using Rust."""
        if not self.available:
            # Fallback logic handled by Factory/Arbiter, but for safety:
            from src.math_kernel.black_scholes import BlackScholesEngine

            return BlackScholesEngine().price(params, option_type)

        return self.core.black_scholes_price(
            params.S, params.K, params.T, params.sigma, params.r, params.q, option_type == "call"
        )

    def price_batch(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """High-performance batch pricing."""
        if not self.available:
            from src.math_kernel.black_scholes import BlackScholesEngine

            return BlackScholesEngine().price_batch(S, K, T, sigma, r, q, option_types)

        is_call = np.where(option_types == "call", True, False)
        return self.core.batch_black_scholes(S, K, T, sigma, r, q, is_call)

    def greeks(self, params: BSParameters, option_type: str = "call") -> dict[str, float]:
        """Compute Greeks using Rust."""
        if not self.available:
            from src.math_kernel.black_scholes import BlackScholesEngine

            return BlackScholesEngine().greeks(params, option_type)

        d, g, t, v, r = self.core.black_scholes_greeks(
            params.S, params.K, params.T, params.sigma, params.r, params.q, option_type == "call"
        )
        return {"delta": d, "gamma": g, "theta": t, "vega": v, "rho": r}

    def simulate_gbm(
        self, S0: np.ndarray, mu: np.ndarray, sigma: np.ndarray, T: float, dt: float, steps: int
    ) -> np.ndarray:
        """High-order RK4 GBM simulation using Rust."""
        if not self.available:
            from src.math_kernel.gbm_solver import GBMSolver

            return GBMSolver().simulate(S0, mu, sigma, T, dt)

        return self.core.runge_kutta_4_gbm(S0, mu, sigma, T, dt, steps, None)
