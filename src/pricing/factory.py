"""
Pricing Engine Factory (Refactored)
==================================

Implements a hardware-aware Strategy Pattern for option pricing.
Supports dynamic registration and execution strategy selection (JIT, WASM, GPU).
"""


import structlog

from src.pricing.base import BasePricingEngine


class PricingEngineNotFound(Exception):
    """Custom exception raised when a requested pricing engine is not found."""
    pass

logger = structlog.get_logger(__name__)

class PricingEngineFactory:
    """
    Centralized registry for pricing engines.
    Automatically selects the optimal execution strategy based on available hardware.
    """

    _engines: dict[str, type[BasePricingEngine]] = {}
    _instances: dict[str, BasePricingEngine] = {}
    _default_engine_override: str | None = None

    @classmethod
    def set_default_engine(cls, name: str | None):
        """Dynamic override for the default engine (used for AIOps model switching)."""
        cls._default_engine_override = name.lower() if name else None
        logger.warning("pricing_factory_override_set", engine=name)

    @classmethod
    def register(cls, name: str, engine_cls: type[BasePricingEngine]):
        """Registers a pricing engine class."""
        cls._engines[name.lower()] = engine_cls
        logger.info("pricing_engine_registered", engine=name)

    @classmethod
    def get_engine(cls, name: str, execution_strategy: str | None = None) -> BasePricingEngine:
        """
        Get an engine instance. 
        Execution strategy can be forced (e.g., 'wasm', 'jit', 'gpu').
        """
        # Apply global override if set and no specific strategy is forced
        if cls._default_engine_override and execution_strategy is None:
            name = cls._default_engine_override

        name = name.lower()
        
        # Check if we should override with WASM
        from src.pricing.wasm_engine import WASM_AVAILABLE
        if execution_strategy == "wasm" or (WASM_AVAILABLE and execution_strategy is None and name in ["heston", "monte_carlo"]):
            name = "wasm"

        if name in cls._instances:
            return cls._instances[name]

        if name not in cls._engines:
            # Fallback to lazy loading for legacy support
            cls._lazy_load(name)

        if name not in cls._engines:
            raise PricingEngineNotFound(f"Unknown pricing engine: {name}")

        engine_cls = cls._engines[name]
        instance = engine_cls()
        cls._instances[name] = instance
        return instance

    @classmethod
    def _lazy_load(cls, name: str):
        """Lazy load core engines to prevent circular imports."""
        try:
            if name == "black_scholes":
                from src.pricing.black_scholes import BlackScholesEngine
                cls.register("black_scholes", BlackScholesEngine)
            elif name == "monte_carlo":
                from src.pricing.monte_carlo import MonteCarloEngine
                cls.register("monte_carlo", MonteCarloEngine)
            elif name == "wasm":
                from src.pricing.wasm_engine import WASMPricingEngine
                cls.register("wasm", WASMPricingEngine)
            elif name == "neural":
                from src.ml.models.neural_engine import NeuralPricingEngine
                cls.register("neural", NeuralPricingEngine)
            # Add more as needed
        except ImportError as e:
            logger.error("lazy_load_failed", engine=name, error=str(e))

# Auto-initialize with core engines
PricingEngineFactory._lazy_load("black_scholes")
PricingEngineFactory._lazy_load("monte_carlo")