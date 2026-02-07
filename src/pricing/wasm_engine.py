from typing import Any

import numpy as np
import structlog

from .base import PricingStrategy
from .models import BSParameters, OptionGreeks

logger = structlog.get_logger()

try:
    # This assumes the wasm module is built and available in the python path
    # Typically via a wrapper like wasmer or by calling a node process,
    # or if we've built a python extension using the same rust code.
    # For this architecture, we'll assume a 'wasm_loader' utility exists.
    from src.utils.wasm_loader import get_wasm_instance

    WASM_AVAILABLE = True
except ImportError:
    WASM_AVAILABLE = False
    logger.warning("wasm_engine_unavailable", reason="wasm_loader_not_found")


class WASMPricingEngine(PricingStrategy):
    """
    High-performance bridge to Rust/WASM pricing implementation.
    Optimized for large batch processing.
    """

    def __init__(self, model: str = "black_scholes"):
        self.instance = get_wasm_instance() if WASM_AVAILABLE else None
        self.model = model.lower()

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        if not self.instance:
            return 0.0  # Fallback should be handled by factory

        # SOTA: Route to specialized WASM solvers based on model type (Task 3)
        if self.model in ["monte_carlo", "mc"]:
            return self.price_monte_carlo(params, option_type)
        elif self.model in ["fdm", "crank_nicolson"]:
            return self.price_american_cn(params, option_type)
        elif self.model == "heston":
            # Heston requires specific parameters not in BSParameters directly
            # This would typically come from a symbol lookup, handled in PricingService
            return 0.0

        # Default to Black-Scholes
        if option_type == "call":
            return self.instance.price_call(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            )
        else:
            return self.instance.price_put(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            )

    def calculate_greeks(
        self, params: BSParameters, option_type: str = "call"
    ) -> OptionGreeks:
        if not self.instance:
            raise RuntimeError("WASM instance not available")

        res = self.instance.calculate_greeks(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
        )
        return OptionGreeks(
            delta=res.delta,
            gamma=res.gamma,
            vega=res.vega,
            theta=res.theta,
            rho=res.rho,
        )

    def batch_price_black_scholes(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        is_call: np.ndarray,
    ) -> np.ndarray:
        """
        Processes a batch of options using WASM SIMD acceleration.
        """
        if not self.instance:
            return np.array([])

        # Pack into Float64Array-compatible flat buffer for WASM
        # Stride = 7: spot, strike, time, vol, rate, div, is_call
        num_options = len(S)
        input_data = np.zeros(num_options * 7, dtype=np.float64)
        for i in range(num_options):
            off = i * 7
            input_data[off] = S[i]
            input_data[off + 1] = K[i]
            input_data[off + 2] = T[i]
            input_data[off + 3] = sigma[i]
            input_data[off + 4] = r[i]
            input_data[off + 5] = q[i]
            input_data[off + 6] = 1.0 if is_call[i] else 0.0

        # Call WASM batch function (returns flat array of results)
        raw_results = self.instance.batch_calculate_simd(input_data)

        # Extract prices (stride 6 in results: price, delta, gamma, vega, theta, rho)
        return raw_results[::6]

    def price_american_lsm(
        self,
        params: BSParameters,
        option_type: str = "call",
        num_paths: int = 10000,
        num_steps: int = 50,
    ) -> float:
        """Access high-speed Rust LSM implementation."""
        if not self.instance:
            return 0.0

        return self.instance.price_american_lsm(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            num_paths,
            num_steps,
        )

    def price_monte_carlo(
        self, params: BSParameters, option_type: str = "call", num_paths: int = 100000
    ) -> float:
        """Rust/WASM Monte Carlo implementation."""
        if not self.instance:
            return 0.0
        return self.instance.price_monte_carlo(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            num_paths,
        )

    def price_american_cn(
        self,
        params: BSParameters,
        option_type: str = "call",
        m: int = 200,
        n: int = 200,
    ) -> float:
        """Rust/WASM Crank-Nicolson implementation."""
        if not self.instance:
            return 0.0
        return self.instance.price_american(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            m,
            n,
        )

    def price_heston(
        self, params: Any, spot: float, strike: float, time: float, r: float
    ) -> float:
        """Rust/WASM Heston implementation."""
        if not self.instance:
            return 0.0

        return self.instance.price_heston(
            spot,
            strike,
            time,
            r,
            params.v0,
            params.kappa,
            params.theta,
            params.sigma,
            params.rho,
        )

    def price_heston_mc(
        self,
        params: Any,
        spot: float,
        strike: float,
        time: float,
        r: float,
        option_type: str = "call",
        num_paths: int = 10000,
    ) -> float:
        """Rust/WASM Heston Monte Carlo implementation."""
        if not self.instance:
            return 0.0
        return self.instance.price_heston_mc(
            spot,
            strike,
            time,
            r,
            params.v0,
            params.kappa,
            params.theta,
            params.sigma,
            params.rho,
            option_type == "call",
            num_paths,
        )

    def batch_price_heston(
        self,
        spot: np.ndarray,
        strike: np.ndarray,
        time: np.ndarray,
        r: np.ndarray,
        params: Any,
    ) -> np.ndarray:
        """Rust/WASM Heston batch implementation."""
        if not self.instance:
            return np.array([])

        # Stride = 9: spot, strike, time, r, v0, kappa, theta, sigma, rho
        num_options = len(spot)
        input_data = np.zeros(num_options * 9, dtype=np.float64)
        for i in range(num_options):
            off = i * 9
            input_data[off] = spot[i]
            input_data[off + 1] = strike[i]
            input_data[off + 2] = time[i]
            input_data[off + 3] = r[i]
            input_data[off + 4] = params.v0
            input_data[off + 5] = params.kappa
            input_data[off + 6] = params.theta
            input_data[off + 7] = params.sigma
            input_data[off + 8] = params.rho

        return self.instance.batch_price_heston(input_data)

    def batch_price_monte_carlo(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        is_call: np.ndarray,
        num_paths: int = 100000,
    ) -> np.ndarray:
        """Rust/WASM Monte Carlo batch implementation."""
        if not self.instance:
            return np.array([])

        num_options = len(S)
        input_data = np.zeros(num_options * 7, dtype=np.float64)
        for i in range(num_options):
            off = i * 7
            input_data[off] = S[i]
            input_data[off + 1] = K[i]
            input_data[off + 2] = T[i]
            input_data[off + 3] = sigma[i]
            input_data[off + 4] = r[i]
            input_data[off + 5] = q[i]
            input_data[off + 6] = 1.0 if is_call[i] else 0.0

        return self.instance.batch_price_monte_carlo(input_data, num_paths)

    def batch_price_american_cn(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        is_call: np.ndarray,
        m: int = 200,
        n: int = 200,
    ) -> np.ndarray:
        """Rust/WASM Crank-Nicolson batch implementation."""
        if not self.instance:
            return np.array([])

        num_options = len(S)
        input_data = np.zeros(num_options * 7, dtype=np.float64)
        for i in range(num_options):
            off = i * 7
            input_data[off] = S[i]
            input_data[off + 1] = K[i]
            input_data[off + 2] = T[i]
            input_data[off + 3] = sigma[i]
            input_data[off + 4] = r[i]
            input_data[off + 5] = q[i]
            input_data[off + 6] = 1.0 if is_call[i] else 0.0

        return self.instance.batch_price_american(input_data, m, n)
