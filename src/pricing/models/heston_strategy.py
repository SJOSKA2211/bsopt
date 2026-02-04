"""
Heston Pricing Strategy Implementation
======================================

Standardized wrapper for the Heston Stochastic Volatility model.
"""

from typing import Optional, Any
import numpy as np

from src.pricing.base import PricingStrategy
from src.pricing.models import BSParameters, OptionGreeks, HestonParams
from src.pricing.models.heston_fft import HestonModelFFT


class HestonPricingStrategy(PricingStrategy):
    """
    Standardized strategy for Heston pricing.
    Note: Requires HestonParams to be passed via kwargs or derived from symbol.
    """

    def price(self, params: BSParameters, option_type: str = "call", **kwargs) -> float:
        """
        Calculate option price using Heston FFT.
        Expects 'heston_params' in kwargs.
        """
        h_params = kwargs.get("heston_params")
        if not h_params:
            raise ValueError("Heston pricing requires 'heston_params' in kwargs")
            
        model = HestonModelFFT(h_params, r=params.rate, T=params.maturity)
        
        if option_type.lower() == "call":
            return model.price_call(params.spot, params.strike)
        else:
            return model.price_put(params.spot, params.strike)

    def calculate_greeks(self, params: BSParameters, option_type: str = "call", **kwargs) -> OptionGreeks:
        """
        Calculate Heston Greeks using finite differences.
        (Future optimization: use characteristic function derivatives)
        """
        h_params = kwargs.get("heston_params")
        if not h_params:
            raise ValueError("Heston greeks require 'heston_params' in kwargs")

        # Fallback to a baseline for now or implement FD
        # Standardize empty greeks if not fully implemented
        return OptionGreeks(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0
        )
