"""
Black-Scholes-Merton Option Pricing Engine

This module provides the core mathematical implementation for pricing
European options and calculating their sensitivity measures (Greeks).
"""

from dataclasses import asdict
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np

from .base import PricingStrategy
from .models import BSParameters, OptionGreeks
from .quant_utils import batch_bs_price_jit, batch_greeks_jit


class BlackScholesEngine(PricingStrategy, VectorizedPricingStrategy): # Added VectorizedPricingStrategy
    """
    Unified pricing engine for European options.
    Supports both single option pricing and high-performance batch pricing.
    """

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Implementation of PricingStrategy interface."""
        if option_type.lower() == "call":
            return self.price_call(params)
        return self.price_put(params)

    @classmethod
    def calculate_greeks(cls, params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> OptionGreeks:
        """
        Implementation of PricingStrategy interface.
        Supports both (params, option_type) and keyword arguments.
        """
        if isinstance(params, BSParameters):
            data = asdict(params)
        else:
            data = kwargs
        
        if "option_type" not in data:
            data["option_type"] = option_type

        S, K, T, sigma, r, q, is_call = cls._prepare_batch(
            data.get("spot"),
            data.get("strike"),
            data.get("maturity"),
            data.get("volatility"),
            data.get("rate"),
            data.get("dividend"),
            data.get("option_type", "call"),
        )

        delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)

        return OptionGreeks(float(delta[0]), float(gamma[0]), float(vega[0]), float(theta[0]), float(rho[0]))

    @staticmethod
    def calculate_d1_d2(params: BSParameters) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        Assumes BSParameters contains scalar (float) values.
        """
        spot = params.spot
        strike = params.strike
        maturity = params.maturity
        volatility = params.volatility
        rate = params.rate
        dividend = params.dividend

        # Handle zero maturity edge case
        if maturity <= 1e-12:
            if spot > strike:
                return 100.0, 100.0  # Approximations for in-the-money
            elif spot < strike:
                return -100.0, -100.0  # Approximations for out-of-the-money
            else:
                return 0.0, 0.0  # At-the-money

        d1 = (
            np.log(spot / strike)
            + (rate - dividend + 0.5 * volatility**2) * maturity
        ) / (volatility * np.sqrt(maturity))

        d2 = d1 - volatility * np.sqrt(maturity)

        return float(d1), float(d2)

    @staticmethod
    def _prepare_batch(spot, strike, maturity, volatility, rate, dividend, option_type):
        S = np.atleast_1d(np.asarray(spot, dtype=np.float64))
        K = np.atleast_1d(np.asarray(strike, dtype=np.float64))
        T = np.atleast_1d(np.asarray(maturity, dtype=np.float64))
        sigma = np.atleast_1d(np.asarray(volatility, dtype=np.float64))
        r = np.atleast_1d(np.asarray(rate, dtype=np.float64))
        q = np.atleast_1d(np.asarray(dividend, dtype=np.float64))

        if isinstance(option_type, str):
            if option_type.lower() not in ["call", "put"]:
                raise ValueError("option_type must be 'call' or 'put'")
            is_call = np.full(len(S), option_type.lower() == "call", dtype=bool)
        elif isinstance(option_type, (list, np.ndarray)):
            types = np.asarray(option_type)
            is_call = np.array([str(t).lower() == "call" for t in types], dtype=bool)
        else:
            is_call = np.atleast_1d(np.asarray(option_type, dtype=bool))

        return np.broadcast_arrays(S, K, T, sigma, r, q, is_call)

    @classmethod
    def price_options(cls, params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> np.ndarray: # Always returns np.ndarray
        """
        Generic pricing method for one or many options.
        """
        # Determine if the original input was a single scalar for post-processing
        is_original_single_input = False
        if params is not None:
            # If params object is given, assume it holds scalar (float) values as per BSParameters definition
            is_original_single_input = True
        elif 'spot' in kwargs and np.isscalar(kwargs.get('spot')):
            # If keyword arguments are given, check if spot is scalar (assuming other params follow)
            is_original_single_input = True

        if isinstance(params, BSParameters):
            data = asdict(params)
            data["option_type"] = option_type
        else:
            data = kwargs
            if "option_type" not in data:
                data["option_type"] = option_type

        S, K, T, sigma, r, q, is_call = cls._prepare_batch(
            data.get("spot"),
            data.get("strike"),
            data.get("maturity"),
            data.get("volatility"),
            data.get("rate"),
            data.get("dividend"),
            data.get("option_type"),
        )
        prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
        
        # If original input was a single scalar, return a scalar for convenience
        if is_original_single_input and prices.size == 1:
            return prices.item()
        return prices

    @classmethod
    def calculate_greeks_batch(
        cls,
        params: Optional[Union[BSParameters, float, np.ndarray]] = None,
        option_type: Union[str, np.ndarray] = "call",
        **kwargs,
    ) -> Union[OptionGreeks, Dict[str, np.ndarray]]:
        """
        Generic Greek calculation for one or many options.
        Supports both (params, option_type) and keyword arguments.
        """
        # Determine if the original input was a single scalar for post-processing
        is_original_single_input = False
        if isinstance(params, BSParameters):
            is_original_single_input = True
        elif params is not None and np.isscalar(params):
            is_original_single_input = True
        elif 'spot' in kwargs and np.isscalar(kwargs.get('spot')):
            is_original_single_input = True

        if isinstance(params, BSParameters):
            data = asdict(params)
            data["option_type"] = option_type
        elif params is not None:
            data = {"spot": params, "option_type": option_type} # Here params is a scalar spot
            data.update(kwargs)
        else:
            data = kwargs

        S, K, T, sigma, r, q, is_call = cls._prepare_batch(
            data.get("spot"),
            data.get("strike"),
            data.get("maturity"),
            data.get("volatility"),
            data.get("rate"),
            data.get("dividend"),
            data.get("option_type", "call"),
        )

        delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)

        if is_original_single_input and delta.size == 1:
            return OptionGreeks(float(delta[0]), float(gamma[0]), float(vega[0]), float(theta[0]), float(rho[0]))

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    @classmethod
    def price_call(cls, params: BSParameters) -> float:
        """Compatibility method for BSParameters."""
        return float(
            cls.price_options(
                spot=params.spot,
                strike=params.strike,
                maturity=params.maturity,
                volatility=params.volatility,
                rate=params.rate,
                dividend=params.dividend,
                option_type="call",
            )
        )

    @classmethod
    def price_put(cls, params: BSParameters) -> float:
        """Compatibility method for BSParameters."""
        return float(
            cls.price_options(
                spot=params.spot,
                strike=params.strike,
                maturity=params.maturity,
                volatility=params.volatility,
                rate=params.rate,
                dividend=params.dividend,
                option_type="put",
            )
        )

    # Implement price_single for VectorizedPricingStrategy interface
    def price_single(self, params: BSParameters, option_type: str = "call") -> float:
        """Calculate the price of a single option using the vectorized engine."""
        return float(self.price_options(params=params, option_type=option_type))

    # Implement price_batch for VectorizedPricingStrategy interface
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
        return batch_bs_price_jit(S, K, T, sigma, r, q, is_call)


def verify_put_call_parity(params: BSParameters, tolerance: float = 1e-10) -> bool:
    """
    Verify if put-call parity holds for the given parameters.
    """
    call_price = BlackScholesEngine.price_call(params)
    put_price = BlackScholesEngine.price_put(params)

    lhs = call_price - put_price
    rhs = params.spot * np.exp(-params.dividend * params.maturity) - params.strike * np.exp(
        -params.rate * params.maturity
    )

    return bool(abs(lhs - rhs) < tolerance)
