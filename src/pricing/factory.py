"""
Pricing Engine Factory
======================

Provides a centralized way to create pricing strategy instances
based on model type.
"""

from typing import Dict, Type, Any
from src.utils.lazy_import import lazy_import
import sys

# Lazy import map for strategies
_STRATEGY_MAP = {
    "black_scholes": ".black_scholes.BlackScholesEngine",
    "monte_carlo": ".monte_carlo.MonteCarloEngine",
    "binomial": ".lattice.BinomialTreePricer",
    "fdm": ".finite_difference.CrankNicolsonSolver",
}

from src.utils.circuit_breaker import pricing_circuit

class PricingEngineFactory:
    """Factory for creating pricing strategies using lazy loading and instance caching."""

    _instances: Dict[str, Any] = {}

    @classmethod
    @pricing_circuit
    def get_strategy(cls, model_type: str) -> Any:
        """
        Get a pricing strategy instance for the given model type.
        Uses a cache to return the same instance for multiple calls.
        """
        model_key = model_type.lower()
        
        # Return cached instance if available
        if model_key in cls._instances:
            return cls._instances[model_key]

        if model_key not in _STRATEGY_MAP:
            raise ValueError(f"Unknown pricing model: {model_type}")

        # Extract module and class name
        path = _STRATEGY_MAP[model_key]
        module_path, class_name = path.rsplit('.', 1)
        
        # Use lazy_import utility
        strategy_class = lazy_import(
            "src.pricing",
            {class_name: module_path},
            class_name,
            sys.modules[__name__]
        )

        instance = strategy_class()
        cls._instances[model_key] = instance
        return instance
