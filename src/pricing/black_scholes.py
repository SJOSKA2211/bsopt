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


class BlackScholesEngine(PricingStrategy):
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
        """
        # Handle array inputs by taking first element or converting
        # Use .item() for scalars or first element for arrays
        spot = float(params.spot.item() if params.spot.size == 1 else params.spot[0])
        strike = float(params.strike.item() if params.strike.size == 1 else params.strike[0])
        maturity = float(params.maturity.item() if params.maturity.size == 1 else params.maturity[0])
        volatility = float(params.volatility.item() if params.volatility.size == 1 else params.volatility[0])
        rate = float(params.rate.item() if params.rate.size == 1 else params.rate[0])
        dividend = float(params.dividend.item() if params.dividend.size == 1 else params.dividend[0])

        # Handle zero maturity edge case
        if maturity <= 1e-12:
            if spot > strike:
                return 100.0, 100.0
            elif spot < strike:
                return -100.0, -100.0
            else:
                return 0.0, 0.0

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
    def price_options(cls, params: Optional[BSParameters] = None, option_type: str = "call", **kwargs) -> Union[float, np.ndarray]:
        """
        Generic pricing method for one or many options.
        """
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

        # Determine if we should return a scalar or array
        # Check if the result has size 1 and if original spot/strike in kwargs or params were not sequences
        # BSParameters converts them to at least 1d or 0d arrays.
        
        is_single = True
        spot_val = data.get("spot")
        strike_val = data.get("strike")
        
        if isinstance(spot_val, (list, np.ndarray)) and np.asanyarray(spot_val).size > 1:
            is_single = False
        if isinstance(strike_val, (list, np.ndarray)) and np.asanyarray(strike_val).size > 1:
            is_single = False

        if is_single and prices.size == 1:
            return float(prices.item())
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
        if isinstance(params, BSParameters):
            data = asdict(params)
            data["option_type"] = option_type
        elif params is not None:
            data = {"spot": params, "option_type": option_type}
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

        is_single = not isinstance(data.get("spot"), (np.ndarray, list)) and not isinstance(
            data.get("strike"), (np.ndarray, list)
        )

        if is_single and delta.size == 1:
            return OptionGreeks(delta[0], gamma[0], vega[0], theta[0], rho[0])

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
