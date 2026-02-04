"""
Core Pricing Engine Interfaces
==============================

Defines standard interfaces for all pricing models to ensure modularity,
testability, and consistent patterns across the platform.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from src.pricing.models import BSParameters, OptionGreeks


class PricingStrategy(ABC):
    """Abstract base class for all pricing strategies."""

    @abstractmethod
    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Calculate the option price."""
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
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        sigma: np.ndarray,
        r: np.ndarray,
        q: np.ndarray,
        is_call: np.ndarray,
    ) -> np.ndarray:
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

    def __init__(self, strategy: Union[PricingStrategy, VectorizedPricingStrategy]):
        self.strategy = strategy

    def get_price(self, params: BSParameters, option_type: str = "call") -> float:
        """Unified entry point for single option pricing."""
        if isinstance(self.strategy, PricingStrategy):
            return self.strategy.price(params, option_type)
        elif isinstance(self.strategy, VectorizedPricingStrategy):
            # If strategy is VectorizedPricingStrategy, use its price_single method
            return self.strategy.price_single(params, option_type)
        else:
            raise TypeError("Unsupported pricing strategy type")
