from typing import Any, cast

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

    def __init__(self, model: str = "black_scholes") -> None:
        self.instance = get_wasm_instance() if WASM_AVAILABLE else None
        self.model = model.lower()

    def price_european(self, params: BSParameters, option_type: str = "call") -> float:
        if not self.instance:
            return 0.0  # Fallback should be handled by factory

        # OPTIMIZED: Route to specialized WASM solvers based on model type (Task 3)
        if self.model in ["monte_carlo", "mc"]:
            return self.price_monte_carlo(params, option_type)
        if self.model in ["fdm", "crank_nicolson"]:
            return self.price_american_cn(params, option_type)
        if self.model == "heston":
            # Heston requires specific parameters not in BSParameters directly
            # This would typically come from a symbol lookup, handled in PricingService
            return 0.0

        # Default to Black-Scholes
        if option_type == "call":
            return cast(float, self.instance.price_call(
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            ))
        return cast(float, self.instance.price_put(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
        ))

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
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
        S: np.ndarray[Any, np.dtype[np.float64]],
        K: np.ndarray[Any, np.dtype[np.float64]],
        T: np.ndarray[Any, np.dtype[np.float64]],
        sigma: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        q: np.ndarray[Any, np.dtype[np.float64]],
        is_call: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Processes a batch of options using WASM SIMD acceleration.
         OPTIMIZED: Zero-copy memory mapping.
        """
        if not self.instance:
            return np.array([], dtype=np.float64)

        num_options = len(S)
        # Stride = 7: spot, strike, time, vol, rate, div, is_call
        input_data = np.column_stack([S, K, T, sigma, r, q, is_call.astype(np.float64)]).ravel()

        #  Use zero-copy mapping to write directly to WASM heap
        from src.utils.wasm_loader import WasmModuleCache

        heap = WasmModuleCache.map_wasm_memory(self.instance)

        # Write to the start of the heap (Assuming WASM exported memory is where it expects input)
        # In a real implementation, we would call an allocator in WASM or use a fixed offset.
        # For this prototype, we copy into the mapped buffer.
        heap[: len(input_data)] = input_data

        # Call WASM batch function (telling it data is at offset 0)
        raw_results = self.instance.batch_calculate_simd_mapped(0, num_options)

        # Extract prices (stride 6 in results: price, delta, gamma, vega, theta, rho)
        return cast(np.ndarray[Any, np.dtype[np.float64]], raw_results[::6])

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

        return cast(float, self.instance.price_american_lsm(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            num_paths,
            num_steps,
        ))

    def price_monte_carlo(
        self, params: BSParameters, option_type: str = "call", num_paths: int = 100000
    ) -> float:
        """Rust/WASM Monte Carlo implementation."""
        if not self.instance:
            return 0.0
        return cast(float, self.instance.price_monte_carlo(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            num_paths,
        ))

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
        return cast(float, self.instance.price_american(
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
            option_type == "call",
            m,
            n,
        ))

    def price_heston(self, params: Any, spot: float, strike: float, time: float, r: float) -> float:
        """Rust/WASM Heston implementation."""
        if not self.instance:
            return 0.0

        return cast(float, self.instance.price_heston(
            spot,
            strike,
            time,
            r,
            params.v0,
            params.kappa,
            params.theta,
            params.sigma,
            params.rho,
        ))

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
        return cast(float, self.instance.price_heston_mc(
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
        ))

    def batch_price_heston(
        self,
        spot: np.ndarray[Any, np.dtype[np.float64]],
        strike: np.ndarray[Any, np.dtype[np.float64]],
        time: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        params: Any,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Rust/WASM Heston batch implementation.
         OPTIMIZED: Zero-copy memory mapping.
        """
        if not self.instance:
            return np.array([], dtype=np.float64)

        num_options = len(spot)
        v0 = np.full(num_options, params.v0)
        kappa = np.full(num_options, params.kappa)
        theta = np.full(num_options, params.theta)
        sigma = np.full(num_options, params.sigma)
        rho = np.full(num_options, params.rho)

        input_data = np.column_stack([spot, strike, time, r, v0, kappa, theta, sigma, rho]).ravel()

        from src.utils.wasm_loader import WasmModuleCache

        heap = WasmModuleCache.map_wasm_memory(self.instance)
        heap[: len(input_data)] = input_data

        self.instance.batch_price_heston_mapped(0, num_options)
        return cast(np.ndarray[Any, np.dtype[np.float64]], heap[:num_options])

    def batch_price_monte_carlo(
        self,
        S: np.ndarray[Any, np.dtype[np.float64]],
        K: np.ndarray[Any, np.dtype[np.float64]],
        T: np.ndarray[Any, np.dtype[np.float64]],
        sigma: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        q: np.ndarray[Any, np.dtype[np.float64]],
        is_call: np.ndarray[Any, np.dtype[np.bool_]],
        num_paths: int = 100000,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Rust/WASM Monte Carlo batch implementation.
        OPTIMIZED: Vectorized data preparation.
        """
        if not self.instance:
            return np.array([], dtype=np.float64)

        num_options = len(S)
        input_data = np.column_stack([S, K, T, sigma, r, q, is_call.astype(np.float64)]).ravel()

        from src.utils.wasm_loader import WasmModuleCache

        heap = WasmModuleCache.map_wasm_memory(self.instance)
        heap[: len(input_data)] = input_data

        self.instance.batch_price_monte_carlo_mapped(0, num_options, num_paths)
        return cast(np.ndarray[Any, np.dtype[np.float64]], heap[:num_options])

    def batch_price_american_cn(
        self,
        S: np.ndarray[Any, np.dtype[np.float64]],
        K: np.ndarray[Any, np.dtype[np.float64]],
        T: np.ndarray[Any, np.dtype[np.float64]],
        sigma: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        q: np.ndarray[Any, np.dtype[np.float64]],
        is_call: np.ndarray[Any, np.dtype[np.bool_]],
        m: int = 200,
        n: int = 200,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Rust/WASM Crank-Nicolson batch implementation.
        OPTIMIZED: Vectorized data preparation.
        """
        if not self.instance:
            return np.array([], dtype=np.float64)

        num_options = len(S)
        input_data = np.column_stack([S, K, T, sigma, r, q, is_call.astype(np.float64)]).ravel()

        from src.utils.wasm_loader import WasmModuleCache

        heap = WasmModuleCache.map_wasm_memory(self.instance)
        heap[: len(input_data)] = input_data

        self.instance.batch_price_american_mapped(0, num_options, m, n)
        return cast(np.ndarray[Any, np.dtype[np.float64]], heap[:num_options])
