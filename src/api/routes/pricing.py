"""
Pricing Routes
==============

RESTful API endpoints for options pricing and Greeks calculation.
"""

import logging
import time
from typing import cast

from fastapi import APIRouter, HTTPException

from src.api.schemas.pricing import (
    BatchPriceRequest,
    BatchPriceResponse,
    GreeksRequest,
    GreeksResponse,
    ImpliedVolatilityRequest,
    ImpliedVolatilityResponse,
    PriceRequest,
    PriceResponse,
)
from src.pricing.black_scholes import BSParameters, OptionGreeks
from src.pricing.factory import PricingEngineFactory
from src.pricing.implied_vol import implied_volatility

router = APIRouter(prefix="/pricing", tags=["pricing"])
logger = logging.getLogger(__name__)


@router.post("/price", response_model=PriceResponse)
async def calculate_price(request: PriceRequest):
    """
    Calculate the price of a single option.
    """
    start_time = time.perf_counter()
    try:
        params = BSParameters(
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            volatility=request.volatility,
            rate=request.rate,
            dividend=request.dividend_yield,
        )

        strategy = PricingEngineFactory.get_strategy(request.model)
        price = strategy.price(params, request.option_type)

        computation_time = (time.perf_counter() - start_time) * 1000

        return PriceResponse(
            price=price,
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            rate=request.rate,
            volatility=request.volatility,
            option_type=request.option_type,
            model=request.model,
            cached=False,
            computation_time_ms=computation_time,
        )
    except Exception as e:
        logger.error(f"Error calculating price with {request.model}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch", response_model=BatchPriceResponse)
async def calculate_batch_prices(request: BatchPriceRequest):
    """
    Calculate prices for multiple options in a single request.
    """
    start_time = time.perf_counter()
    results = []

    for opt_req in request.options:
        try:
            params = BSParameters(
                spot=opt_req.spot,
                strike=opt_req.strike,
                maturity=opt_req.time_to_expiry,
                volatility=opt_req.volatility,
                rate=opt_req.rate,
                dividend=opt_req.dividend_yield,
            )

            strategy = PricingEngineFactory.get_strategy(opt_req.model)
            price = strategy.price(params, opt_req.option_type)

            results.append(
                PriceResponse(
                    price=price,
                    spot=opt_req.spot,
                    strike=opt_req.strike,
                    time_to_expiry=opt_req.time_to_expiry,
                    rate=opt_req.rate,
                    volatility=opt_req.volatility,
                    option_type=opt_req.option_type,
                    model=opt_req.model,
                    cached=False,
                    computation_time_ms=0.0,  # Individual time not tracked in loop for brevity
                )
            )
        except Exception as e:
            logger.warning(f"Error in batch calculation for one option: {e}")
            # We could choose to fail the whole batch or return an error indicator
            # For now, let's just skip it or raise error if critical
            continue

    computation_time = (time.perf_counter() - start_time) * 1000

    return BatchPriceResponse(
        results=results,
        total_count=len(results),
        computation_time_ms=computation_time,
        cached_count=0,
    )


@router.post("/greeks", response_model=GreeksResponse)
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho).
    """
    try:
        params = BSParameters(
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            volatility=request.volatility,
            rate=request.rate,
            dividend=request.dividend_yield,
        )

        # We don't have a model field in GreeksRequest yet, defaulting to black_scholes
        strategy = PricingEngineFactory.get_strategy("black_scholes")
        greeks_res = strategy.calculate_greeks(params, request.option_type)
        greeks = cast(OptionGreeks, greeks_res)

        price = strategy.price(params, request.option_type)

        return GreeksResponse(
            delta=float(greeks.delta),
            gamma=float(greeks.gamma),
            theta=float(greeks.theta),
            vega=float(greeks.vega),
            rho=float(greeks.rho),
            option_price=price,
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            volatility=request.volatility,
            option_type=request.option_type,
        )
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/implied-volatility", response_model=ImpliedVolatilityResponse)
async def calculate_iv(request: ImpliedVolatilityRequest):
    """
    Calculate implied volatility from a market price.
    """
    try:
        iv = implied_volatility(
            market_price=request.option_price,
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            rate=request.rate,
            dividend=request.dividend_yield,
            option_type=request.option_type,
        )

        return ImpliedVolatilityResponse(
            implied_volatility=iv,
            option_price=request.option_price,
            spot=request.spot,
            strike=request.strike,
            iterations=0,  # Not currently returned by implementation
            converged=True,
        )
    except Exception as e:
        logger.error(f"Error calculating Implied Volatility: {e}")
        raise HTTPException(status_code=400, detail=str(e))
