from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
import structlog
import numpy as np

from src.pricing.models import BSParameters, OptionGreeks
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.monte_carlo import MonteCarloEngine, MCConfig
from src.pricing.wasm_engine import WASMPricingEngine
from src.config import settings

logger = structlog.get_logger(__name__)

class PricingModel(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    WASM = "wasm"
    QUANTUM = "quantum"

@dataclass
class PricingRequest:
    params: BSParameters
    option_type: str = "call"
    model: Optional[PricingModel] = None
    engine_config: Optional[Dict[str, Any]] = None
    style: str = "european" # european, american
    use_gpu: bool = False

class EngineArbiter:
    """
    Intelligent routing logic to select the optimal pricing engine
    based on request characteristics, system load, and configuration.
    """
    def __init__(self):
        self.bs_engine = BlackScholesEngine()
        self.wasm_engine = WASMPricingEngine()
        self.default_mc_engine = MonteCarloEngine() 

    def route_request(self, request: PricingRequest) -> float:
        """
        Routes the pricing request to the optimal engine.
        """
        logger.debug("routing_pricing_request", 
                     model=request.model, 
                     style=request.style, 
                     use_gpu=request.use_gpu)

        # 1. Explicit Model Selection
        if request.model == PricingModel.WASM:
             if self.wasm_engine.instance:
                 if request.style == "american":
                     return self.wasm_engine.price_american_lsm(request.params, request.option_type)
                 return self.wasm_engine.price(request.params, request.option_type)
             else:
                 logger.warning("wasm_requested_but_unavailable_fallback_bs")
                 return self.bs_engine.price(request.params, request.option_type)

        if request.model == PricingModel.MONTE_CARLO:
            config = MCConfig(**request.engine_config) if request.engine_config else None
            engine = MonteCarloEngine(config)
            if request.style == "american":
                return engine.price_american_lsm(request.params, request.option_type)
            return engine.price(request.params, request.option_type)

        if request.model == PricingModel.BLACK_SCHOLES:
            return self.bs_engine.price(request.params, request.option_type)

        # 2. Automatic Arbitration (Smart Routing)
        
        # American Options -> Prefer WASM (Speed) > Monte Carlo (Accuracy/Fallback)
        if request.style == "american":
            if self.wasm_engine.instance:
                return self.wasm_engine.price_american_lsm(request.params, request.option_type)
            return self.default_mc_engine.price_american_lsm(request.params, request.option_type)

        # European Options
        # For standard European options, Python's vectorized/JIT BS is highly optimized.
        # However, if we needed to offload CPU from the Python process, WASM would be a good choice.
        # For now, we stick to the robust BS Engine.
        return self.bs_engine.price(request.params, request.option_type)

    def route_batch(self, 
                    S: np.ndarray, 
                    K: np.ndarray, 
                    T: np.ndarray, 
                    sigma: np.ndarray, 
                    r: np.ndarray, 
                    is_call: np.ndarray,
                    model: Optional[PricingModel] = None) -> np.ndarray:
        """
        Routes batch requests efficiently.
        """
        # WASM batch is likely faster for very large batches due to SIMD in Rust
        if model == PricingModel.WASM and self.wasm_engine.instance:
            q = np.zeros_like(S) # Default dividend 0
            return self.wasm_engine.batch_price_black_scholes(S, K, T, sigma, r, q, is_call)
            
        # Default to Python Vectorized BS
        dividend = np.zeros_like(S)
        option_types = np.where(is_call == 1, "call", "put")
        return self.bs_engine.price_batch(S, K, T, sigma, r, dividend, option_types)
