"""
Core Pricing Engine Interfaces

Defines standard interfaces for all pricing models to ensure modularity,
testability, and consistent patterns across the platform.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.math_kernel.models import BSParameters, OptionGreeks


class PricingStrategy(ABC):
    """Abstract base class for all pricing strategies."""

    @abstractmethod
    def price_european(self, params: BSParameters, option_type: str = "call") -> Any:
        """Calculate the European option price."""
        pass

    @abstractmethod
    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Calculate option sensitivity measures."""
        pass


BasePricingEngine = PricingStrategy


class VectorizedPricingStrategy(ABC):
    """Abstract base class for high-performance vectorized pricing."""

    @abstractmethod
    def price_batch(
        self,
        S: np.ndarray[Any, np.dtype[np.float64]],
        K: np.ndarray[Any, np.dtype[np.float64]],
        T: np.ndarray[Any, np.dtype[np.float64]],
        sigma: np.ndarray[Any, np.dtype[np.float64]],
        r: np.ndarray[Any, np.dtype[np.float64]],
        q: np.ndarray[Any, np.dtype[np.float64]],
        is_call: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Batch calculation of option prices."""
        pass

    @abstractmethod
    def price_single(self, params: BSParameters, option_type: str = "call") -> float:
        """Calculate the price of a single option."""
        pass


class PricingEngine:
    """
    Standardized Pricing Engine using the Strategy Pattern.
    Refactored for performance and modularity (Ultrathinking).
    """

    def __init__(self, strategy: PricingStrategy | VectorizedPricingStrategy) -> None:
        self.strategy = strategy

    def get_price(self, params: BSParameters, option_type: str = "call") -> float:
        """Unified entry point for single option pricing."""
        if isinstance(self.strategy, PricingStrategy):
            return float(self.strategy.price_european(params, option_type))
        if isinstance(self.strategy, VectorizedPricingStrategy):
            # If strategy is VectorizedPricingStrategy, use its price_single method
            return self.strategy.price_single(params, option_type)
        raise TypeError("Unsupported pricing strategy type")