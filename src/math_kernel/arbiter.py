from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import structlog

from src.math_kernel.models import BSParameters

logger = structlog.get_logger(__name__)


class PricingModel(StrEnum):
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    WASM = "wasm"
    QUANTUM = "quantum"
    RUST = "rust"


@dataclass
class PricingRequest:
    params: BSParameters
    option_type: str = "call"
    model: PricingModel | None = None
    engine_config: dict[str, Any] | None = None
    style: str = "european"  # european, american
    use_gpu: bool = False


class EngineArbiter:
    """
    Intelligent routing logic to select the optimal pricing engine.
    OPTIMIZED: Respects AIOps overrides via PricingEngineFactory.
    """

    def __init__(self):
        from src.math_kernel.factory import PricingEngineFactory

        self.factory = PricingEngineFactory

    def route_request(self, request: PricingRequest) -> float:
        """
        Routes the pricing request to the optimal engine.
        """
        # 1. Resolve Engine via Factory (Handles dynamic AIOps overrides)
        strategy = None
        if request.model == PricingModel.WASM:
            strategy = "wasm"
        elif request.model == PricingModel.RUST:
            strategy = "rust"

        engine = self.factory.get_engine(
            request.model or "black_scholes", execution_strategy=strategy
        )

        logger.debug(
            "routing_pricing_request",
            model=engine.__class__.__name__,
            style=request.style,
        )

        if request.style == "american":
            return engine.price_american_lsm(request.params, request.option_type)

        return engine.price(request.params, request.option_type)

    def route_batch(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        is_call: np.ndarray,
        model: PricingModel | None = None,
    ) -> np.ndarray:
        """
        Routes batch requests efficiently with smart defaulting.
        """
        # OPTIMIZED: Respect explicit model choice, otherwise auto-select based on size
        if model is None:
            if len(S) > 10000:
                engine_name = "rust"
            elif len(S) > 1000:
                engine_name = "wasm"
            else:
                engine_name = "black_scholes"
        else:
            engine_name = str(model)

        # High-speed Rust path for massive CPU-parallel batches
        if engine_name == "rust":
            try:
                import equaflow_core
                q = np.zeros_like(S)
                return equaflow_core.batch_black_scholes(S, K, T, sigma, r, q, is_call.astype(bool))
            except ImportError:
                logger.warning("rust_core_not_available_falling_back")
                engine_name = "black_scholes"

        engine = self.factory.get_engine(engine_name)

        # High-speed WASM path for Black-Scholes batches
        if engine_name == "wasm" and hasattr(engine, "batch_price_black_scholes"):
            q = np.zeros_like(S)
            return engine.batch_price_black_scholes(S, K, T, sigma, r, q, is_call)

        # Standard vectorized path
        dividend = np.zeros_like(S)
        option_types = np.where(is_call == 1, "call", "put")
        return engine.price_batch(S, K, T, sigma, r, dividend, option_types)
