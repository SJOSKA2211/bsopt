"""
Pricing Tasks for Celery - Production Optimized
"""

import asyncio
import gc
import time
from typing import Any, cast

import msgspec
import numpy as np
import structlog

from services.pricing.implied_vol import implied_volatility
from services.pricing.models import BSParameters
from services.shared.math_utils import (
    calculate_greeks,
    calculate_greeks_scalar,
    calculate_price,
    calculate_price_scalar,
)
from services.utils.cache import pricing_cache
from services.utils.celery import BaseAsyncTask
from services.utils.distributed import RayOrchestrator
from services.utils.ray_pool import RayActorPool

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


# Initialize Ray once
RayOrchestrator.init()

# Initialize Global Pool for Math Actors
_math_pool: RayActorPool | None = None


def get_math_pool() -> RayActorPool:
    global _math_pool
    if _math_pool is None:
        from services.workers.ray_workers import MathActor

        _math_pool = RayActorPool(MathActor, name="math_v2")
    return _math_pool


@celery_app.task(base=BaseAsyncTask, bind=True, queue="pricing", max_retries=3)
def recalibrate_symbol_task(self: BaseAsyncTask, symbol: str) -> dict[str, Any]:
    """Non-blocking calibration delegation using Ray Actor Pool."""
    try:
        results = self.run_async(_recalibrate_symbols_batch_impl([symbol]))
        return cast(dict[str, Any], results[0])
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e) from e


@celery_app.task(base=BaseAsyncTask, bind=True, queue="pricing")
def recalibrate_symbols_batch_task(self: BaseAsyncTask, symbols: list[str]) -> list[dict[str, Any]]:
    """HIGH-PERFORMANCE: Batch calibration delegation to Ray."""
    try:
        return cast(list[dict[str, Any]], self.run_async(_recalibrate_symbols_batch_impl(symbols)))
    except Exception as e:
        logger.error("batch_calibration_task_failed", symbols=symbols, error=str(e))
        return []


async def _recalibrate_symbols_batch_impl(symbols: list[str]) -> list[dict[str, Any]]:
    """Async implementation of batch calibration using Ray Actor Pool."""
    from services.data.router import MarketDataRouter

    try:
        # 1. Fetch market data snapshots in parallel
        router = MarketDataRouter()
        snapshots = await asyncio.gather(*[router.get_option_chain_snapshot(s) for s in symbols])

        valid_symbols = []
        valid_data = []
        for s, data in zip(symbols, snapshots):
            if data:
                valid_symbols.append(s)
                valid_data.append(data)

        if not valid_symbols:
            return []

        # 2. Delegate to Ray Actor Pool
        pool = get_math_pool()
        actor = await pool.get_actor()

        #  HIGH-PERFORMANCE: Non-blocking await of Ray future
        return await actor.run_calibration_batch.remote(valid_symbols, valid_data)

    except Exception as exc:
        logger.error("batch_calib_impl_error", symbols=symbols, error=str(exc))
        return []


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
    self: PricingTask,
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
                    "price": round(float(cached_price), 4),
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
    self: PricingTask,
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
            prices = cast(
                np.ndarray[Any, np.dtype[np.float64]],
                calculate_price(spots, strikes, maturities, vols, rates, divs, is_call),
            )

            # Perform vectorized greeks using JIT utilities
            deltas, gammas, thetas, vegas, rhos = cast(
                tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
                calculate_greeks(spots, strikes, maturities, vols, rates, divs, is_call),
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
            result_list = cast(list[dict[str, Any]], msgspec.to_builtins(results))
        else:
            # Fallback to sequential pricing using scalar JIT functions
            result_list = []
            for opt in options:
                is_call_bool = opt.get("option_type", "call").lower() == "call"
                price = calculate_price_scalar(
                    opt["spot"],
                    opt["strike"],
                    opt["maturity"],
                    opt["volatility"],
                    opt["rate"],
                    opt.get("dividend", 0.0),
                    is_call_bool,
                )
                delta, gamma, theta, vega, rho = calculate_greeks_scalar(
                    opt["spot"],
                    opt["strike"],
                    opt["maturity"],
                    opt["volatility"],
                    opt["rate"],
                    opt.get("dividend", 0.0),
                    is_call_bool,
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
    self: PricingTask,
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
    base=PricingTask,
    queue="pricing",
)
def generate_volatility_surface_task(
    self: PricingTask,
    prices: list[list[float]],
    strikes: list[float],
    maturities: list[float],
    spot: float,
    rate: float,
    option_type: str = "call",
) -> dict[str, Any]:
    """
    Generate a full volatility surface from a grid of prices.
    Uses SABR model for calibration of each maturity slice.
    """
    from services.pricing.vol_surface import CalibrationEngine, MarketQuote, VolatilitySurface

    logger.info("vol_surface_gen_start", option_type=option_type, n_maturities=len(maturities))

    engine = CalibrationEngine()
    surface = VolatilitySurface()

    results = {}

    # 1. Calibrate each maturity slice
    for i, t in enumerate(maturities):
        quotes = []
        for j, k in enumerate(strikes):
            price = prices[i][j]
            if price <= 0:
                continue

            # Calculate IV for this point first
            iv = implied_volatility(price, spot, k, t, rate, 0.0, option_type)
            if iv > 0:
                quotes.append(
                    MarketQuote(
                        strike=k, maturity=t, implied_vol=iv, forward=spot * np.exp(rate * t)
                    )
                )

        if len(quotes) >= 3:
            try:
                # Use SABR for high-fidelity surface
                sabr_params, _ = engine.calibrate_sabr(quotes)
                from services.pricing.vol_surface import SABRModel

                surface.add_slice(t, SABRModel(sabr_params), quotes[0].forward)
                results[str(t)] = {
                    "alpha": sabr_params.alpha,
                    "beta": sabr_params.beta,
                    "rho": sabr_params.rho,
                    "nu": sabr_params.nu,
                }
            except Exception as e:
                logger.warning("slice_calibration_failed", maturity=t, error=str(e))

    return {
        "status": "completed",
        "maturities": maturities,
        "strikes": strikes,
        "calibrated_slices": results,
        "timestamp": time.time(),
    }
