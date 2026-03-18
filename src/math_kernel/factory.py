"""
Pricing Engine Factory (Refactored)

Implements a hardware-aware Strategy Pattern for option pricing.
Supports dynamic registration and execution strategy selection (JIT, WASM, GPU).
"""

import structlog

from src.math_kernel.base import BasePricingEngine


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
        from src.math_kernel.wasm_engine import WASM_AVAILABLE

        if execution_strategy == "wasm" or (
            WASM_AVAILABLE and execution_strategy is None and name in ["heston", "monte_carlo"]
        ):
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
        """Lazy load src.shared engines to prevent circular imports."""
        try:
            if name == "black_scholes":
                from src.math_kernel.black_scholes import BlackScholesEngine

                cls.register("black_scholes", BlackScholesEngine)
            elif name == "monte_carlo":
                from src.math_kernel.monte_carlo import MonteCarloEngine

                cls.register("monte_carlo", MonteCarloEngine)
            elif name == "wasm":
                from src.math_kernel.wasm_engine import WASMPricingEngine

                cls.register("wasm", WASMPricingEngine)
            elif name == "neural":
                from src.ml.models.neural_engine import NeuralPricingEngine

                cls.register("neural", NeuralPricingEngine)
            elif name == "lattice":
                from src.math_kernel.lattice import LatticePricingEngine

                cls.register("lattice", LatticePricingEngine)
            elif name == "fdm":
                from src.math_kernel.finite_difference import FDMPricingEngine

                cls.register("fdm", FDMPricingEngine)
            elif name == "exotic":
                from src.math_kernel.exotic import ExoticPricingEngine

                cls.register("exotic", ExoticPricingEngine)
            elif name == "heston":
                from src.math_kernel.models.heston_strategy import HestonPricingStrategy

                cls.register("heston", HestonPricingStrategy)
            # Add more as needed
        except ImportError as e:
            logger.error("lazy_load_failed", engine=name, error=str(e))


# Auto-initialize with src.shared engines
PricingEngineFactory._lazy_load("black_scholes")
PricingEngineFactory._lazy_load("monte_carlo")
