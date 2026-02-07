import numpy as np

from src.pricing.models import BSParameters, OptionGreeks

from .base import PricingStrategy
from .quant_utils import (
    batch_bs_price_jit,
    batch_greeks_jit,
    scalar_bs_price_jit,
    scalar_greeks_jit,
)


class BlackScholesEngine(PricingStrategy):
    """
    Vectorized Black-Scholes-Merton engine using Numba JIT kernels.
    Supports broad-casted array operations for multi-option pricing.
    """

    def price(
        self, params: BSParameters | None = None, option_type: str = "call", **kwargs
    ) -> float | np.ndarray:
        return self.price_options(params=params, option_type=option_type, **kwargs)

    @staticmethod
    def calculate_greeks(
        params: BSParameters | None = None, option_type: str = "call", **kwargs
    ) -> OptionGreeks | dict[str, np.ndarray]:
        return BlackScholesEngine.calculate_greeks_batch(
            params=params, option_type=option_type, **kwargs
        )

    @staticmethod
    def price_options(
        spot: float | np.ndarray | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: BSParameters | None = None,
        out: np.ndarray | None = None,
        **kwargs
    ) -> float | np.ndarray:
        if params is not None:
            spot, strike, maturity, volatility, rate, dividend = (
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            )

        if spot is None:
            raise ValueError("Missing spot price")

        if np.isscalar(spot) and np.isscalar(strike) and isinstance(option_type, str):
            return scalar_bs_price_jit(
                float(spot),
                float(strike),
                float(maturity),
                float(volatility),
                float(rate),
                float(dividend),
                option_type.lower() == "call",
            )

        S, K, T, sigma, r, q = np.broadcast_arrays(
            np.atleast_1d(spot),
            np.atleast_1d(strike),
            np.atleast_1d(maturity),
            np.atleast_1d(volatility),
            np.atleast_1d(rate),
            np.atleast_1d(dividend),
        )

        is_call_scalar = (
            (option_type.lower() == "call")
            if isinstance(option_type, str)
            else np.array([s.lower() == "call" for s in option_type], dtype=bool)
        )

        if np.isscalar(is_call_scalar):
            is_call = np.full(S.shape, is_call_scalar, dtype=bool)
        else:
            is_call = is_call_scalar

        result = batch_bs_price_jit(
            S.astype(np.float64),
            K.astype(np.float64),
            T.astype(np.float64),
            sigma.astype(np.float64),
            r.astype(np.float64),
            q.astype(np.float64),
            is_call,
            out=out,
        )

        return result[0] if np.isscalar(spot) and result.size == 1 else result

    @staticmethod
    def calculate_greeks_batch(
        spot: float | np.ndarray | None = None,
        strike: float | np.ndarray | None = None,
        maturity: float | np.ndarray | None = None,
        volatility: float | np.ndarray | None = None,
        rate: float | np.ndarray | None = None,
        dividend: float | np.ndarray = 0.0,
        option_type: str | np.ndarray = "call",
        params: BSParameters | None = None,
        **kwargs
    ) -> OptionGreeks | dict[str, np.ndarray]:
        if params is not None:
            spot, strike, maturity, volatility, rate, dividend = (
                params.spot,
                params.strike,
                params.maturity,
                params.volatility,
                params.rate,
                params.dividend,
            )

        if spot is None:
            raise ValueError("Missing spot price")

        if np.isscalar(spot) and isinstance(option_type, str):
            delta, gamma, vega, theta, rho = scalar_greeks_jit(
                float(spot),
                float(strike),
                float(maturity),
                float(volatility),
                float(rate),
                float(dividend),
                option_type.lower() == "call",
            )
            return OptionGreeks(
                delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho
            )

        S, K, T, sigma, r, q = np.broadcast_arrays(
            np.atleast_1d(spot),
            np.atleast_1d(strike),
            np.atleast_1d(maturity),
            np.atleast_1d(volatility),
            np.atleast_1d(rate),
            np.atleast_1d(dividend),
        )

        is_call = (
            (option_type.lower() == "call")
            if isinstance(option_type, str)
            else np.array([s.lower() == "call" for s in option_type], dtype=bool)
        )

        delta, gamma, vega, theta, rho = batch_greeks_jit(
            S.astype(np.float64),
            K.astype(np.float64),
            T.astype(np.float64),
            sigma.astype(np.float64),
            r.astype(np.float64),
            q.astype(np.float64),
            is_call,
        )

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    @staticmethod
    def verify_put_call_parity(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        call_price: float,
        put_price: float,
        dividend: float = 0.0,
    ) -> bool:
        """Verify Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)"""
        lhs = call_price - put_price
        rhs = spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity)
        return np.isclose(lhs, rhs, atol=1e-4)


def verify_put_call_parity(*args, **kwargs):
    """Module-level wrapper for Put-Call Parity verification."""
    return BlackScholesEngine.verify_put_call_parity(*args, **kwargs)
