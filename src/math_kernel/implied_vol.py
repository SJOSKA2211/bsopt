"""
Optimized Implied Volatility Calculator

Features:
- Corrado-Miller initial guess for faster convergence
- Vectorized Newton-Raphson with adaptive step size
- Robust error handling with Brent fallback
"""

from typing import cast

import numpy as np

from src.math_kernel.quant_utils import (
    corrado_miller_initial_guess,
    vectorized_newton_raphson_iv_jit,
)
from src.shared.math_utils import calculate_greeks, calculate_price

try:
    import bsopt_core

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

class ImpliedVolatilityError(Exception):
    """Exception raised when IV calculation fails to converge."""

    pass

def _calculate_intrinsic_value(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    is_call: bool,
) -> float:
    """Calculate the discounted intrinsic value of an option."""
    if is_call:
        return float(
            max(
                spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity),
                0.0,
            )
        )
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
    is_call: bool,
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

    intrinsic = _calculate_intrinsic_value(spot, strike, rate, dividend, maturity, is_call)
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
    is_call: bool,
    initial_guess: float = 0.25,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    OPTIMIZED: Scalar Newton-Raphson using vectorized kernels (fallback to scalar).
    """
    sigma = initial_guess

    for _ in range(max_iterations):
        # 1. Price using vectorized kernel (handles scalar via numpy)
        price = calculate_price(spot, strike, maturity, sigma, rate, dividend, is_call)

        # 2. Vega using vectorized kernel
        _, _, _, vega, _ = calculate_greeks(spot, strike, maturity, sigma, rate, dividend, is_call)

        diff = price - market_price
        if abs(diff) < tolerance:
            return float(sigma)

        # Check for vanishing vega (avoid div by zero)
        if abs(vega) < 1e-12:
            break

        # 3. Newton-Raphson update
        sigma -= diff / (vega * 100.0)
        sigma = max(1e-6, min(sigma, 5.0))

    raise ImpliedVolatilityError("failed to converge")

def _brent_iv(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    is_call: bool,
    tolerance: float = 1e-8,
) -> float:
    """Brent's method for IV fallback."""
    from scipy.optimize import brentq

    def obj(sigma):
        price_val = calculate_price(spot, strike, maturity, sigma, rate, dividend, is_call)
        return float(price_val) - market_price

    try:
        return float(brentq(obj, 1e-6, 5.0, xtol=tolerance))
    except Exception:
        raise ImpliedVolatilityError("failed to converge") from None

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
    method: str = "auto",
    initial_guess: float | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Calculate IV for a single option using specified method."""
    if method not in ["auto", "newton", "brent"]:
        raise ValueError("method must be 'auto', 'newton', or 'brent'")

    is_call = option_type.lower() == "call"
    _validate_inputs(market_price, spot, strike, maturity, rate, dividend, is_call)

    if initial_guess is None:
        initial_guess = corrado_miller_initial_guess(
            np.array([market_price]),
            np.array([spot]),
            np.array([strike]),
            np.array([maturity]),
            np.array([rate]),
            np.array([dividend]),
            np.array([0 if is_call else 1]),
        )[0]

    if method == "brent":
        return _brent_iv(market_price, spot, strike, maturity, rate, dividend, is_call, tolerance)

    # Default/Newton
    try:
        return _newton_raphson_iv(
            market_price,
            spot,
            strike,
            maturity,
            rate,
            dividend,
            is_call,
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
                is_call,
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
    
    is_call = np.char.lower(option_types.astype(str)) == "call"

    if CORE_AVAILABLE:
        try:
            return bsopt_core.batch_black_scholes_iv(
                market_prices.astype(np.float64),
                spots.astype(np.float64),
                strikes.astype(np.float64),
                maturities.astype(np.float64),
                rates.astype(np.float64),
                dividends.astype(np.float64),
                is_call.astype(bool),
                tolerance,
                max_iterations,
            )
        except Exception as e:
            import structlog

            structlog.get_logger().warning("core_batch_iv_failed_fallback_to_jit", error=str(e))

    # 2. Corrado-Miller Initial Guess
    # 0 for call, 1 for put in corrado_miller_initial_guess
    type_ints = np.where(is_call, 0, 1)
    sigma_guess = corrado_miller_initial_guess(
        market_prices.astype(np.float64),
        spots.astype(np.float64),
        strikes.astype(np.float64),
        maturities.astype(np.float64),
        rates.astype(np.float64),
        dividends.astype(np.float64),
        type_ints,
    )

    # 3. Optimized JIT Newton-Raphson
    sigma = vectorized_newton_raphson_iv_jit(
        market_prices.astype(np.float64),
        spots.astype(np.float64),
        strikes.astype(np.float64),
        maturities.astype(np.float64),
        rates.astype(np.float64),
        dividends.astype(np.float64),
        is_call.astype(bool),
        sigma_guess,
        tolerance,
        max_iterations,
    )

    return cast(np.ndarray, sigma)
