"""
Optimized Implied Volatility Calculator

Features:
- Corrado-Miller initial guess for faster convergence
- Vectorized Newton-Raphson with adaptive step size
- Robust error handling
"""

from typing import cast

import numpy as np

from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.quant_utils import (
    corrado_miller_initial_guess,
    vectorized_newton_raphson_iv_jit,
)


class ImpliedVolatilityError(Exception):
    pass


def _calculate_intrinsic_value(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    option_type: str,
) -> float:
    """Calculate the discounted intrinsic value of an option."""
    if option_type.lower() == "call":
        return float(
            max(
                spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity),
                0.0,
            )
        )
    else:
        return float(
            max(
                strike * np.exp(-rate * maturity) - spot * np.exp(-dividend * maturity),
                0.0,
            )
        )


def _validate_inputs(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: str,
) -> None:
    """Validate inputs for IV calculation."""
    if market_price < 0:
        raise ValueError("market_price cannot be negative")
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if option_type.lower() not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    intrinsic = _calculate_intrinsic_value(
        spot, strike, rate, dividend, maturity, option_type
    )
    if market_price < intrinsic - 1e-7:
        raise ValueError(
            f"Arbitrage violation: market price {market_price} is below intrinsic value {intrinsic}"
        )

    if market_price < 1e-12:
        raise ImpliedVolatilityError("market price too close to zero")


def _newton_raphson_iv(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: str,
    initial_guess: float = 0.25,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Single-option Newton-Raphson implementation."""
    if initial_guess <= 0:
        raise ValueError("initial_guess must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    sigma = initial_guess
    for _ in range(max_iterations):
        results = BlackScholesEngine.calculate_greeks_batch(
            spot=np.array([spot]),
            strike=np.array([strike]),
            maturity=np.array([maturity]),
            volatility=np.array([sigma]),
            rate=np.array([rate]),
            dividend=np.array([dividend]),
            option_type=np.array([option_type]),
        )
        price_res = BlackScholesEngine.price_options(
            spot=np.array([spot]),
            strike=np.array([strike]),
            maturity=np.array([maturity]),
            volatility=np.array([sigma]),
            rate=np.array([rate]),
            dividend=np.array([dividend]),
            option_type=np.array([option_type]),
        )
        price = (
            float(price_res[0])
            if isinstance(price_res, np.ndarray)
            else float(price_res)
        )

        vega = cast(np.ndarray, results["vega"])[0] * 100.0
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        if abs(vega) < 1e-12:
            break

        sigma -= diff / vega
        sigma = max(1e-6, min(sigma, 5.0))

    raise ImpliedVolatilityError("failed to converge")


def _brent_iv(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: str,
    tolerance: float = 1e-8,
) -> float:
    """Brent's method for IV (placeholder for tests)."""
    from scipy.optimize import brentq

    def obj(sigma):
        price_res = BlackScholesEngine.price_options(
            spot=np.array([spot]),
            strike=np.array([strike]),
            maturity=np.array([maturity]),
            volatility=np.array([sigma]),
            rate=np.array([rate]),
            dividend=np.array([dividend]),
            option_type=np.array([option_type]),
        )
        price_val = (
            float(price_res[0])
            if isinstance(price_res, np.ndarray)
            else float(price_res)
        )
        return price_val - market_price

    try:
        return float(brentq(obj, 1e-6, 5.0, xtol=tolerance))
    except Exception:
        raise ImpliedVolatilityError("failed to converge")


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
    method: str = "auto",
    initial_guess: float = 0.25,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Calculate IV for a single option using specified method."""
    if method not in ["auto", "newton", "brent"]:
        raise ValueError("method must be 'auto', 'newton', or 'brent'")
    _validate_inputs(market_price, spot, strike, maturity, rate, dividend, option_type)

    if method == "brent":
        return _brent_iv(
            market_price, spot, strike, maturity, rate, dividend, option_type, tolerance
        )

    # Default/Newton
    try:
        return _newton_raphson_iv(
            market_price,
            spot,
            strike,
            maturity,
            rate,
            dividend,
            option_type,
            initial_guess,
            tolerance,
            max_iterations,
        )
    except ImpliedVolatilityError:
        # Fallback to Brent if Newton fails and auto
        if method == "auto":
            return _brent_iv(
                market_price,
                spot,
                strike,
                maturity,
                rate,
                dividend,
                option_type,
                tolerance,
            )
        raise


def vectorized_implied_volatility(
    market_prices: np.ndarray,
    spots: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rates: np.ndarray,
    dividends: np.ndarray,
    option_types: np.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
) -> np.ndarray:
    """
    State-of-the-art vectorized IV calculation.
    """
    # 1. Convert types to bool for JIT (True=call, False=put)
    is_call = np.array([str(t).lower() == "call" for t in option_types], dtype=bool)
    type_ints = np.where(is_call, 0, 1)

    # 2. Corrado-Miller Initial Guess
    sigma = corrado_miller_initial_guess(
        market_prices, spots, strikes, maturities, rates, dividends, type_ints
    )

    # 3. Optimized JIT Newton-Raphson
    sigma = vectorized_newton_raphson_iv_jit(
        market_prices,
        spots,
        strikes,
        maturities,
        rates,
        dividends,
        is_call,
        sigma,
        tolerance,
        max_iterations,
    )

    return cast(np.ndarray, sigma)
