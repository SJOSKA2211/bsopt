"""
Pricing Engine Factory
======================

Provides a centralized way to create pricing strategy instances
based on model type.
"""

from typing import Dict, Type

from src.pricing.base import PricingStrategy
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.lattice import BinomialTreePricer
from src.pricing.monte_carlo import MonteCarloEngine
from src.utils.circuit_breaker import pricing_circuit


class PricingEngineFactory:
    """Factory for creating pricing strategies."""

    _strategies: Dict[str, Type[PricingStrategy]] = {
        "black_scholes": BlackScholesEngine,
        "monte_carlo": MonteCarloEngine,
        "binomial": BinomialTreePricer,
        "fdm": CrankNicolsonSolver,
    }

    @classmethod
    @pricing_circuit
    def get_strategy(cls, model_type: str) -> PricingStrategy:
        """
        Get a pricing strategy instance for the given model type.

        Args:
            model_type: One of 'black_scholes', 'monte_carlo', 'binomial'

        Returns:
            An instance of PricingStrategy

        Raises:
            ValueError: If model_type is unknown
        """
        strategy_class = cls._strategies.get(model_type.lower())
        if not strategy_class:
            raise ValueError(f"Unknown pricing model: {model_type}")

        return strategy_class()
