"""
Black-Scholes-Merton Option Pricing Engine

This module provides the core mathematical implementation for pricing
European options and calculating their sensitivity measures (Greeks).
"""

from dataclasses import asdict
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np

from .base import PricingStrategy
from .quant_utils import batch_bs_price_jit, batch_greeks_jit
from .models import BSParameters, OptionGreeks


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

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Implementation of PricingStrategy interface.
        """
        data = asdict(params)
        data["option_type"] = option_type

        S, K, T, sigma, r, q, is_call = self._prepare_batch(
            data.get("spot"),
            data.get("strike"),
            data.get("maturity"),
            data.get("volatility"),
            data.get("rate"),
            data.get("dividend", 0.0),
            data.get("option_type", "call"),
        )

        delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)

        return OptionGreeks(delta[0], gamma[0], vega[0], theta[0], rho[0])

    @staticmethod
    def calculate_d1_d2(params: BSParameters) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes formula.
        """
        # Handle zero maturity edge case
        if params.maturity <= 1e-12:
            if params.spot > params.strike:
                return 100.0, 100.0
            elif params.spot < params.strike:
                return -100.0, -100.0
            else:
                return 0.0, 0.0

        d1 = (
            np.log(params.spot / params.strike)
            + (params.rate - params.dividend + 0.5 * params.volatility**2) * params.maturity
        ) / (params.volatility * np.sqrt(params.maturity))

        d2 = d1 - params.volatility * np.sqrt(params.maturity)

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
        else:
            types = np.asarray(option_type)
            is_call = np.array([t.lower() == "call" for t in types], dtype=bool)

        return np.broadcast_arrays(S, K, T, sigma, r, q, is_call)

    @classmethod
    def price_options(cls, **kwargs) -> Union[float, np.ndarray]:
        """
        Generic pricing method for one or many options.
        """
        S, K, T, sigma, r, q, is_call = cls._prepare_batch(
            kwargs.get("spot"),
            kwargs.get("strike"),
            kwargs.get("maturity"),
            kwargs.get("volatility"),
            kwargs.get("rate"),
            kwargs.get("dividend", 0.0),
            kwargs.get("option_type", "call"),
        )
        prices = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)

        # Determine if we should return a scalar or array
        is_single = not isinstance(kwargs.get("spot"), (np.ndarray, list)) and not isinstance(
            kwargs.get("strike"), (np.ndarray, list)
        )

        return cast(
            Union[float, np.ndarray], prices[0] if is_single and prices.size == 1 else prices
        )

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
            data.get("dividend", 0.0),
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