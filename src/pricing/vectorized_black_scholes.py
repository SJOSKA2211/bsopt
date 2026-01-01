"""
DEPRECATED: Vectorized Black-Scholes Engine wrapper.
Use src.pricing.black_scholes.BlackScholesEngine directly.
"""

import warnings
from typing import Dict, Union

import numpy as np

from src.pricing.black_scholes import BlackScholesEngine

warnings.warn(
    "vectorized_black_scholes is deprecated. "
    "Use src.pricing.black_scholes.BlackScholesEngine instead.",
    DeprecationWarning,
    stacklevel=2,
)


class VectorizedBlackScholesEngine:
    """
    Deprecated wrapper for the unified BlackScholesEngine.
    """

    @staticmethod
    def price_options(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        rate: Union[float, np.ndarray],
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        dtype=np.float64,
    ) -> np.ndarray:
        res = BlackScholesEngine.price_options(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
            option_type=option_type,
        )
        return np.atleast_1d(res)

    @staticmethod
    def calculate_greeks(
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        maturity: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        rate: Union[float, np.ndarray],
        dividend: Union[float, np.ndarray] = 0.0,
        option_type: Union[str, np.ndarray] = "call",
        dtype=np.float64,
    ) -> Dict[str, np.ndarray]:
        res = BlackScholesEngine.calculate_greeks_batch(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
            option_type=option_type,
        )
        if isinstance(res, dict):
            return res
        # Convert OptionGreeks to dict of 1D arrays
        return {
            "delta": np.atleast_1d(res.delta),
            "gamma": np.atleast_1d(res.gamma),
            "vega": np.atleast_1d(res.vega),
            "theta": np.atleast_1d(res.theta),
            "rho": np.atleast_1d(res.rho),
        }
