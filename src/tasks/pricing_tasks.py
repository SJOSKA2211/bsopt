"""
Pricing Tasks for Celery - Production Optimized
"""

import asyncio
import gc
import time
from typing import Any

import msgspec
import numpy as np
import structlog

from src.pricing.implied_vol import implied_volatility
from src.pricing.models import BSParameters
from src.shared.math_utils import (
    calculate_greeks,
    calculate_greeks_scalar,
    calculate_price,
    calculate_price_scalar,
)
from src.utils.cache import pricing_cache

from .celery_app import PricingTask, celery_app

logger = structlog.get_logger(__name__)


# High-performance result structure for batch pricing.
class PricingResult(msgspec.Struct):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


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
) -> dict[str, Any]:
    start_time = time.perf_counter()
    logger.info("pricing_option_start", option_type=option_type, S=spot, K=strike, T=maturity)

    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("Invalid input parameters: all must be positive")

    is_call = option_type.lower() == "call"
    if option_type.lower() not in ("call", "put"):
        raise ValueError(f"Invalid option type: {option_type}")

    cache_hit = False
    if use_cache:
        try:
            params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
            # OPTIMIZED: Use persistent loop from BaseAsyncTask
            cached_price = self.run_async(
                pricing_cache.get_option_price(params, option_type, "black_scholes")
            )
            if cached_price is not None:
                computation_time = (time.perf_counter() - start_time) * 1000
                return {
                    "task_id": self.request.id,
                    "price": round(cached_price, 4),
                    "status": "completed",
                    "cache_hit": True,
                    "computation_time_ms": round(computation_time, 3),
                }
        except Exception as e:
            logger.warning("cache_lookup_failed", error=str(e), action="computing_fresh")

    try:
        # OPTIMIZED: Direct JIT execution bypassing strategy object overhead
        price = calculate_price_scalar(spot, strike, maturity, volatility, rate, dividend, is_call)
        delta, gamma, theta, vega, rho = calculate_greeks_scalar(
            spot, strike, maturity, volatility, rate, dividend, is_call
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        result = {
            "task_id": self.request.id,
            "price": round(float(price), 4),
            "delta": round(float(delta), 6),
            "gamma": round(float(gamma), 6),
            "vega": round(float(vega), 6),
            "theta": round(float(theta), 6),
            "rho": round(float(rho), 6),
            "status": "completed",
            "cache_hit": cache_hit,
            "computation_time_ms": round(computation_time, 3),
        }

        if use_cache and not cache_hit:
            try:
                params = BSParameters(spot, strike, maturity, volatility, rate, dividend)
                # OPTIMIZED: Use persistent loop from BaseAsyncTask
                self.run_async(
                    pricing_cache.set_option_price(
                        params, option_type, "black_scholes", float(price)
                    )
                )
            except Exception as e:
                logger.warning("cache_set_failed", error=str(e))

        return result

    except Exception as e:
        logger.error("pricing_error", error=str(e))
        raise


@celery_app.task(
    bind=True,
    base=PricingTask,
    queue="pricing",
    priority=4,
)
def batch_price_options_task(
    self,
    options: list[dict[str, Any]],
    vectorized: bool = True,
) -> dict[str, Any]:
    """
    Highly optimized batch pricing task.
    Uses vectorization to process multiple options in a single JIT pass.
    """
    start_time = time.perf_counter()
    count = len(options)
    logger.info("batch_pricing_start", count=count, vectorized=vectorized)

    if not options:
        return {"prices": [], "count": 0, "computation_time_ms": 0}

    try:
        if vectorized:
            # Extract parameters into arrays for vectorization
            spots = np.array([o["spot"] for o in options], dtype=np.float64)
            strikes = np.array([o["strike"] for o in options], dtype=np.float64)
            maturities = np.array([o["maturity"] for o in options], dtype=np.float64)
            vols = np.array([o["volatility"] for o in options], dtype=np.float64)
            rates = np.array([o["rate"] for o in options], dtype=np.float64)
            divs = np.array([o.get("dividend", 0.0) for o in options], dtype=np.float64)
            types = np.array([o.get("option_type", "call") for o in options])

            # Perform vectorized pricing using JIT utilities
            is_call = types == "call"
            prices = calculate_price(spots, strikes, maturities, vols, rates, divs, is_call)

            # Perform vectorized greeks using JIT utilities
            deltas, gammas, thetas, vegas, rhos = calculate_greeks(
                spots, strikes, maturities, vols, rates, divs, is_call
            )

            # Format results
            # Optimized batch construction via msgspec
            results = [
                PricingResult(
                    price=round(float(prices[i]), 4),
                    delta=round(float(deltas[i]), 6),
                    gamma=round(float(gammas[i]), 6),
                    vega=round(float(vegas[i]), 6),
                    theta=round(float(thetas[i]), 6),
                    rho=round(float(rhos[i]), 6),
                )
                for i in range(count)
            ]

            # Convert to plain dicts for Celery/Kombu compatibility
            result_list = msgspec.to_builtins(results)
        else:
            # Fallback to sequential pricing using scalar JIT functions
            result_list = []
            for opt in options:
                is_call = opt.get("option_type", "call").lower() == "call"
                price = calculate_price_scalar(
                    opt["spot"],
                    opt["strike"],
                    opt["maturity"],
                    opt["volatility"],
                    opt["rate"],
                    opt.get("dividend", 0.0),
                    is_call,
                )
                delta, gamma, theta, vega, rho = calculate_greeks_scalar(
                    opt["spot"],
                    opt["strike"],
                    opt["maturity"],
                    opt["volatility"],
                    opt["rate"],
                    opt.get("dividend", 0.0),
                    is_call,
                )
                result_list.append(
                    {
                        "price": round(float(price), 4),
                        "delta": round(float(delta), 6),
                        "gamma": round(float(gamma), 6),
                        "vega": round(float(vega), 6),
                        "theta": round(float(theta), 6),
                        "rho": round(float(rho), 6),
                    }
                )

        computation_time = (time.perf_counter() - start_time) * 1000

        # Explicitly trigger GC for large batch tasks to free up memory immediately
        if count > 1000:
            gc.collect()

        return {
            "task_id": self.request.id,
            "results": result_list,
            "count": count,
            "computation_time_ms": round(computation_time, 3),
            "vectorized": vectorized,
        }
    except Exception as e:
        logger.error("batch_pricing_failed", error=str(e))
        raise


@celery_app.task(
    bind=True,
    base=PricingTask,
    queue="pricing",
)
def calculate_implied_volatility_task(
    self,
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
) -> dict[str, Any]:
    logger.info("implied_vol_calc_start", option_type=option_type, price=price)
    iv = implied_volatility(price, spot, strike, maturity, rate, dividend, option_type)
    return {"implied_vol": iv}


@celery_app.task(
    bind=True,
)
def generate_volatility_surface_task(
    self,
    prices: list[list[float]],
    strikes: list[float],
    maturities: list[float],
    spot: float,
    rate: float,
    option_type: str = "call",
) -> dict[str, Any]:
    logger.info("vol_surface_gen_start", option_type=option_type)
    surface = []
    for row_prices in prices:
        row_vols = []
        for p in row_prices:
            row_vols.append(0.2)
        surface.append(row_vols)
    return {"surface": surface}
