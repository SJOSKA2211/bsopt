"""
Pricing Tasks for Celery - Production Optimized
=================================================

Optimized for:
- High throughput with vectorized batch processing
- Caching integration for repeated calculations
- Circuit breaker pattern for external dependencies
- Comprehensive error handling and retry logic
- Task chaining for complex workflows
"""

import logging
import math
import time
from typing import Any, Dict, List, cast

import numpy as np
from scipy import stats

from .celery_app import PricingTask, celery_app

logger = logging.getLogger(__name__)


from src.pricing.black_scholes import BlackScholesEngine
from .celery_app import PricingTask, celery_app

logger = logging.getLogger(__name__)


# =============================================================================
# Pricing Tasks
# =============================================================================


@celery_app.task(
    bind=True,
    base=PricingTask,
    queue="pricing",
    priority=5,
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
)
def price_option_task(
    self,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Async task to price an option using Black-Scholes.

    Args:
        spot: Current stock price
        strike: Strike price
        maturity: Time to maturity in years
        volatility: Annualized volatility
        rate: Risk-free interest rate
        dividend: Dividend yield
        option_type: 'call' or 'put'
        use_cache: Whether to use Redis cache

    Returns:
        dict: Price and Greeks with metadata
    """
    start_time = time.perf_counter()
    logger.info(f"Pricing {option_type} option: S={spot}, K={strike}, T={maturity}")

    # Input validation
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("Invalid input parameters: all must be positive")

    if option_type not in ("call", "put"):
        raise ValueError(f"Invalid option type: {option_type}")

    # Check cache first
    cache_hit = False
    if use_cache:
        try:
            import asyncio

            from src.pricing.black_scholes import BSParameters
            from src.utils.cache import pricing_cache

            params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
            cached_price = asyncio.get_event_loop().run_until_complete(
                pricing_cache.get_option_price(params, option_type, "black_scholes")
            )
            cached_greeks = asyncio.get_event_loop().run_until_complete(
                pricing_cache.get_greeks(params, option_type)
            )

            if cached_price is not None and cached_greeks is not None:
                cache_hit = True
                computation_time = (time.perf_counter() - start_time) * 1000
                return {
                    "task_id": self.request.id,
                    "price": round(cached_price, 4),
                    "delta": cached_greeks.delta,
                    "gamma": cached_greeks.gamma,
                    "vega": cached_greeks.vega,
                    "theta": cached_greeks.theta,
                    "rho": cached_greeks.rho,
                    "status": "completed",
                    "cache_hit": True,
                    "computation_time_ms": round(computation_time, 3),
                }
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}, computing fresh")

    try:
        from src.pricing.black_scholes import BlackScholesEngine, BSParameters

        params = BSParameters(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend
        )

        # Calculate price and greeks using the centralized engine
        price = BlackScholesEngine.price_options(
            params=params,
            option_type=option_type
        )
        
        greeks = BlackScholesEngine.calculate_greeks(
            params=params,
            option_type=option_type
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        result = {
            "task_id": self.request.id,
            "price": round(float(price), 4),
            "delta": round(float(greeks.delta), 6),
            "gamma": round(float(greeks.gamma), 6),
            "vega": round(float(greeks.vega), 6),
            "theta": round(float(greeks.theta), 6),
            "rho": round(float(greeks.rho), 6),
            "status": "completed",
            "cache_hit": cache_hit,
            "computation_time_ms": round(computation_time, 3),
        }

        # Store in cache (fire and forget)
        if use_cache and not cache_hit:
            try:
                import asyncio
                from src.utils.cache import pricing_cache

                loop = asyncio.new_event_loop()
                loop.run_until_complete(
                    pricing_cache.set_option_price(params, option_type, "black_scholes", float(price))
                )
                loop.run_until_complete(pricing_cache.set_greeks(params, option_type, greeks))
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")

        return result

    except Exception as e:
        logger.error(f"Pricing error: {e}")
        raise  # Let Celery handle retry


@celery_app.task(
    bind=True,
    base=PricingTask,
    queue="pricing",
    priority=4,
    time_limit=120,
    soft_time_limit=100,
)
def batch_price_options_task(
    self,
    options: List[Dict[str, Any]],
    use_vectorized: bool = True,
) -> Dict[str, Any]:
    """
    Async task to price multiple options using vectorized NumPy operations.

    Args:
        options: List of option parameter dicts
        use_vectorized: Use vectorized calculation (much faster for large batches)

    Returns:
        dict: Results for all options with timing info
    """
    start_time = time.perf_counter()
    count = len(options)
    logger.info(f"Batch pricing {count} options (vectorized={use_vectorized})")

    if count == 0:
        return {"task_id": self.request.id, "results": [], "count": 0, "computation_time_ms": 0}

    try:
        if use_vectorized and count > 10:
            # Use vectorized calculation for large batches
            spots = [opt["spot"] for opt in options]
            strikes = [opt["strike"] for opt in options]
            maturities = [opt["maturity"] for opt in options]
            volatilities = [opt["volatility"] for opt in options]
            rates = [opt["rate"] for opt in options]
            dividends = [opt.get("dividend", 0.0) for opt in options]
            option_types = [opt.get("option_type", "call") for opt in options]

            prices = BlackScholesEngine.price_options(
                spot=spots,
                strike=strikes,
                maturity=maturities,
                volatility=volatilities,
                rate=rates,
                dividend=dividends,
                option_type=option_types
            )

            # Ensure prices is an ndarray for indexing
            prices_arr = np.atleast_1d(prices)

            results = []
            for i in range(count):
                results.append(
                    {
                        "price": round(float(prices_arr[i]), 4),
                        "status": "completed",
                    }
                )
        else:
            # Use individual calculations for small batches
            results = []
            for opt in options:
                result = price_option_task.apply(
                    args=[
                        opt["spot"],
                        opt["strike"],
                        opt["maturity"],
                        opt["volatility"],
                        opt["rate"],
                        opt.get("dividend", 0.0),
                        opt.get("option_type", "call"),
                    ],
                    kwargs={"use_cache": False},
                )
                results.append(result.get())

        computation_time = (time.perf_counter() - start_time) * 1000

        return {
            "task_id": self.request.id,
            "results": results,
            "count": len(results),
            "computation_time_ms": round(computation_time, 3),
            "throughput_per_sec": (
                round(count / (computation_time / 1000), 1) if computation_time > 0 else 0
            ),
            "vectorized": use_vectorized and count > 10,
        }

    except Exception as e:
        logger.error(f"Batch pricing error: {e}")
        raise


@celery_app.task(
    bind=True,
    base=PricingTask,
    queue="pricing",
    priority=6,
)
def calculate_implied_volatility_task(
    self,
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
    initial_guess: float = 0.3,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Calculate implied volatility using Newton-Raphson method.

    Args:
        market_price: Observed market price of the option
        spot, strike, maturity, rate, dividend: Option parameters
        option_type: 'call' or 'put'
        initial_guess: Starting volatility estimate
        tolerance: Convergence tolerance
        max_iterations: Maximum Newton-Raphson iterations

    Returns:
        dict: Implied volatility and convergence info
    """
    start_time = time.perf_counter()
    logger.info(f"Calculating IV for market_price={market_price}")

    try:
        from src.pricing.black_scholes import BlackScholesEngine
        # Newton-Raphson implementation ... (rest of the task)
        # Note: I'm keeping the original logic but ensure it's functional
        sigma = initial_guess
        iterations = 0

        for i in range(max_iterations):
            iterations = i + 1
            # We need d1, vega, etc. These could be in BlackScholesEngine
            # For now keep as is if it was working or refactor to use engine
            
            from src.pricing.black_scholes import BSParameters
            params = BSParameters(spot, strike, maturity, sigma, rate, dividend)
            
            price = cast(float, BlackScholesEngine.price_options(params, option_type))
            # vega calculation
            import math
            from scipy import stats
            d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * maturity) / (sigma * math.sqrt(maturity))
            vega = spot * math.exp(-dividend * maturity) * stats.norm.pdf(d1) * math.sqrt(maturity)

            if abs(vega) < 1e-10:
                raise ValueError("Vega too small, Newton-Raphson cannot converge")

            error = float(price - market_price)
            if abs(error) < tolerance:
                break

            sigma = sigma - error / vega
            sigma = max(0.001, min(5.0, sigma))  # Bound volatility

        converged = abs(error) < tolerance
        computation_time = (time.perf_counter() - start_time) * 1000

        return {
            "task_id": self.request.id,
            "implied_volatility": round(sigma, 6),
            "converged": converged,
            "iterations": iterations,
            "final_error": round(abs(error), 8),
            "computation_time_ms": round(computation_time, 3),
            "status": "completed" if converged else "warning",
            "message": (
                "Converged" if converged else f"Did not converge within {max_iterations} iterations"
            ),
        }

    except Exception as e:
        logger.error(f"IV calculation error: {e}")
        raise


@celery_app.task(
    bind=True,
    queue="pricing",
    priority=3,
)
def generate_volatility_surface_task(
    self,
    spot: float,
    strikes: List[float],
    maturities: List[float],
    rate: float,
    dividend: float = 0.0,
    base_volatility: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate a volatility surface for visualization and analysis.

    Args:
        spot: Current spot price
        strikes: List of strike prices
        maturities: List of maturities in years
        rate: Risk-free rate
        dividend: Dividend yield
        base_volatility: Base ATM volatility

    Returns:
        dict: 2D volatility surface data
    """
    start_time = time.perf_counter()
    logger.info(f"Generating vol surface: {len(strikes)} strikes x {len(maturities)} maturities")

    try:
        surface = []
        for T in maturities:
            row = []
            for K in strikes:
                # Simple skew model: higher vol for OTM puts
                moneyness = math.log(K / spot)
                skew = -0.1 * moneyness  # Negative skew
                term_effect = 0.05 * (1 - math.exp(-T))  # Term structure
                vol = base_volatility + skew + term_effect
                vol = max(0.05, min(1.5, vol))  # Bound volatility
                row.append(round(vol, 4))
            surface.append(row)

        computation_time = (time.perf_counter() - start_time) * 1000

        return {
            "task_id": self.request.id,
            "surface": surface,
            "strikes": strikes,
            "maturities": maturities,
            "spot": spot,
            "base_volatility": base_volatility,
            "computation_time_ms": round(computation_time, 3),
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Vol surface generation error: {e}")
        raise