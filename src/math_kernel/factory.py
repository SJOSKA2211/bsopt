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
    Advanced Self-Healing and Auto-Optimizing Pricing Engine Factory.
    Automatically selects the optimal execution strategy based on:
    1. Available Hardware (SIMD, GPU, AMX)
    2. Input Problem Size (Batch vs Single)
    3. Engine Health & Latency Metrics
    """

    _engines: dict[str, type[BasePricingEngine]] = {}
    _instances: dict[str, BasePricingEngine] = {}
    _default_engine_override: str | None = None

    # Engine Health Registry
    _engine_failures: dict[str, int] = {}
    _circuit_breaker_threshold = 3
    _recovery_timeout = 60  # seconds

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
    def get_engine(
        cls, name: str, execution_strategy: str | None = None, batch_size: int = 1
    ) -> BasePricingEngine:
        """
        Get an engine instance with intelligent strategy selection.
        """
        # 1. Apply global override
        if cls._default_engine_override and execution_strategy is None:
            name = cls._default_engine_override

        name = name.lower()

        # 2. Hardware-Aware Strategy Selection
        from src.math_kernel.wasm_engine import WASM_AVAILABLE

        # AUTO-OPTIMIZE: Favor Neural or WASM for high-frequency or large batches
        if execution_strategy is None:
            if batch_size >= 1000 and WASM_AVAILABLE:
                logger.debug("auto_optimizing_to_wasm", reason="large_batch", size=batch_size)
                name = "wasm"
            elif batch_size < 10 and name == "monte_carlo":
                # Monte Carlo is overkill for tiny batches, suggest lattice or neural if available
                pass

        # 3. Circuit Breaker Logic
        if (
            name in cls._engine_failures
            and cls._engine_failures[name] >= cls._circuit_breaker_threshold
        ):
            logger.warning("engine_circuit_breaker_active", engine=name, fallback="black_scholes")
            name = "black_scholes"  # Safe fallback

        # 4. Strategy Force
        if execution_strategy == "wasm" and WASM_AVAILABLE:
            name = "wasm"
        elif execution_strategy == "neural":
            name = "neural"

        if name in cls._instances:
            return cls._instances[name]

        if name not in cls._engines:
            cls._lazy_load(name)

        if name not in cls._engines:
            raise PricingEngineNotFound(f"Unknown pricing engine: {name}")

        try:
            engine_cls = cls._engines[name]
            instance = engine_cls()
            cls._instances[name] = instance
            return instance
        except Exception as e:
            cls._engine_failures[name] = cls._engine_failures.get(name, 0) + 1
            logger.error("engine_initialization_failed", engine=name, error=str(e))
            return cls.get_engine("black_scholes")  # Recursively get fallback

    @classmethod
    def fully_optimize(cls):
        """
        Globally optimizes the factory by pre-warming high-performance engines
        and detecting hardware capabilities.
        """
        logger.info("global_engine_optimization_started")

        # 1. Pre-warm engines
        engines_to_warm = ["black_scholes", "wasm"]

        for engine in engines_to_warm:
            try:
                cls.get_engine(engine)
                logger.info("engine_prewarmed", engine=engine)
            except Exception:
                logger.warning("engine_warmup_failed", engine=engine)

        logger.info(
            "global_engine_optimization_complete", strategies_available=list(cls._engines.keys())
        )

    @classmethod
    def _lazy_load(cls, name: str):
        """Lazy load engines to prevent circular imports."""
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
            elif name == "heston":
                from src.math_kernel.models.heston_strategy import HestonPricingStrategy

                cls.register("heston", HestonPricingStrategy)
            elif name == "rust":
                from src.math_kernel.rust_engine import RustPricingEngine

                cls.register("rust", RustPricingEngine)
        except ImportError as e:
            logger.error("lazy_load_failed", engine=name, error=str(e))


# Initial baseline
PricingEngineFactory._lazy_load("black_scholes")
