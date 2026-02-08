from typing import Any

import numpy as np

from src.pricing.models import BSParameters, OptionGreeks
from src.shared.math_utils import calculate_greeks, calculate_price


class BlackScholesEngine:
    """
    Black-Scholes option pricing engine.
    Implements scalar and vectorized calculations for European options using Numba JIT.
    """

    @staticmethod
    def _extract_params(params: Any | None = None, **kwargs) -> tuple:
        """Helper to extract parameters."""
        if params:
            s = getattr(params, 'spot', kwargs.get('spot'))
            k = getattr(params, 'strike', kwargs.get('strike'))
            t = getattr(params, 'maturity', kwargs.get('maturity'))
            v = getattr(params, 'volatility', kwargs.get('volatility'))
            r = getattr(params, 'rate', kwargs.get('rate'))
            d = getattr(params, 'dividend', kwargs.get('dividend', 0.0))
        else:
            s = kwargs.get('spot')
            k = kwargs.get('strike')
            t = kwargs.get('maturity')
            v = kwargs.get('volatility')
            r = kwargs.get('rate')
            d = kwargs.get('dividend', 0.0)

        # Basic validation
        if any(x is None for x in [s, k, t, v, r]):
            raise ValueError("Missing required parameters (spot, strike, maturity, volatility, rate)")

        # Convert to numpy arrays for Numba (float64) - use atleast_1d to avoid Numba NdIter bugs
        return (
            np.atleast_1d(s).astype(np.float64),
            np.atleast_1d(k).astype(np.float64),
            np.atleast_1d(t).astype(np.float64),
            np.atleast_1d(v).astype(np.float64),
            np.atleast_1d(r).astype(np.float64),
            np.atleast_1d(d).astype(np.float64)
        )

    @staticmethod
    def price_options(
        spot: float | np.ndarray | Any | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: Any | None = None
    ) -> float | np.ndarray:
        """
        Calculate European option prices using Black-Scholes formula (JIT Accelerated).
        """
        # If first arg is BSParameters, shift it to params
        if hasattr(spot, 'spot') and params is None:
            params = spot
            spot = None
            if isinstance(strike, (str, np.ndarray)) and option_type == "call":
                option_type = strike
                strike = None

        S, K, T, sigma, r, q = BlackScholesEngine._extract_params(
            params, spot=spot, strike=strike, maturity=maturity, 
            volatility=volatility, rate=rate, dividend=dividend
        )

        # Handle option_type
        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            option_type_arr = np.asanyarray(option_type)
            is_call = np.char.lower(option_type_arr.astype(str)) == "call"

        # 🚀 OPTIMIZATION: Ensure is_call is a numpy boolean array for Numba stability
        is_call_np = np.atleast_1d(is_call).astype(bool)

        prices = calculate_price(S, K, T, sigma, r, q, is_call_np)
        
        # If all inputs were scalars (implied by params object or all floats), return scalar
        is_scalar = False
        if params is not None:
            # Check if any field in BSParameters is a numpy array
            fields = ['spot', 'strike', 'maturity', 'volatility', 'rate', 'dividend']
            is_scalar = all(not isinstance(getattr(params, f, 0.0), np.ndarray) for f in fields)
        elif all(not isinstance(x, np.ndarray) for x in [spot, strike, maturity, volatility, rate, dividend]):
            is_scalar = True
            
        if is_scalar:
            return float(prices[0])
            
        return prices

    @staticmethod
    def calculate_greeks(
        spot: float | np.ndarray | Any | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: Any | None = None
    ) -> OptionGreeks:
        """
        Calculate Greeks for European options (JIT Accelerated).
        """
        # If first arg is BSParameters, shift it to params
        if hasattr(spot, 'spot') and params is None:
            params = spot
            spot = None
            if isinstance(strike, (str, np.ndarray)) and option_type == "call":
                option_type = strike
                strike = None

        S, K, T, sigma, r, q = BlackScholesEngine._extract_params(
            params, spot=spot, strike=strike, maturity=maturity, 
            volatility=volatility, rate=rate, dividend=dividend
        )

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            option_type_arr = np.asanyarray(option_type)
            is_call = np.char.lower(option_type_arr.astype(str)) == "call"

        is_call_np = np.atleast_1d(is_call).astype(bool)

        delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, sigma, r, q, is_call_np)

        is_scalar = False
        if params is not None:
            fields = ['spot', 'strike', 'maturity', 'volatility', 'rate', 'dividend']
            is_scalar = all(not isinstance(getattr(params, f, 0.0), np.ndarray) for f in fields)
        elif all(not isinstance(x, np.ndarray) for x in [spot, strike, maturity, volatility, rate, dividend]):
            is_scalar = True

        if is_scalar:
            return OptionGreeks(
                delta=float(delta[0]), 
                gamma=float(gamma[0]), 
                theta=float(theta[0]), 
                vega=float(vega[0]), 
                rho=float(rho[0])
            )

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    @staticmethod
    def price_call(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="call"))

    @staticmethod
    def price_put(params: BSParameters) -> float:
        return float(BlackScholesEngine.price_options(params=params, option_type="put"))

    @staticmethod
    def price_batch(S, K, T, sigma, r, dividend, option_types) -> np.ndarray:
        """Vectorized pricing returning an array."""
        return BlackScholesEngine.price_options(
            spot=S, strike=K, maturity=T, volatility=sigma, rate=r, dividend=dividend, option_type=option_types
        )

    @staticmethod
    def calculate_greeks_batch(**kwargs) -> dict[str, np.ndarray]:
        """Vectorized Greeks calculation returning a dictionary."""
        greeks = BlackScholesEngine.calculate_greeks(**kwargs)
        return {
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "rho": greeks.rho
        }

    @staticmethod
    def verify_put_call_parity(S, K, T, r, call_price, put_price, q=0.0):
        lhs = np.asanyarray(call_price) - np.asanyarray(put_price)
        rhs = np.asanyarray(S) * np.exp(-np.asanyarray(q) * np.asanyarray(T)) - \
              np.asanyarray(K) * np.exp(-np.asanyarray(r) * np.asanyarray(T))
        return np.allclose(lhs, rhs, atol=1e-5)

    @classmethod
    def price(cls, params: BSParameters | None = None, option_type: str = "call", **kwargs) -> float:
        """Class method for backward compatibility and PricingStrategy interface."""
        return float(cls.price_options(params=params, option_type=option_type, **kwargs))

def black_scholes(*args, **kwargs):
    result = BlackScholesEngine.price_options(*args, **kwargs)
    if len(args) == 5 or 'params' in kwargs:
        return {"price": result}
    return result

def verify_put_call_parity(params_or_S, K=None, T=None, r=None, call_price=None, put_price=None, q=0.0):
    """Module-level parity verifier for test compatibility."""
    if hasattr(params_or_S, 'spot') and K is None:
        p = params_or_S
        cp = BlackScholesEngine.price_options(params=p, option_type="call")
        pp = BlackScholesEngine.price_options(params=p, option_type="put")
        return BlackScholesEngine.verify_put_call_parity(p.spot, p.strike, p.maturity, p.rate, cp, pp, p.dividend)
    return BlackScholesEngine.verify_put_call_parity(params_or_S, K, T, r, call_price, put_price, q)
