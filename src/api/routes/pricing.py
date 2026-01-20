"""
Pricing Routes
==============

RESTful API endpoints for options pricing and Greeks calculation.
"""

import logging
import time
import json
from typing import Any, Dict, Union, cast

from fastapi import APIRouter, Depends, Response
from src.security.auth import get_current_user_flexible
from src.security.rate_limit import rate_limit

from src.api.exceptions import (
    InternalServerException,
    ServiceUnavailableException,
    ValidationException,
)
from src.api.schemas.common import DataResponse, ErrorResponse
from src.utils.cache import pricing_cache, get_redis_client
from src.api.schemas.pricing import (
    BatchPriceRequest,
    BatchPriceResponse,
    GreeksRequest,
    GreeksResponse,
    ImpliedVolatilityRequest,
    ImpliedVolatilityResponse,
    PriceRequest,
    PriceResponse,
    ExoticPriceRequest,
    ExoticPriceResponse,
)
from src.pricing.black_scholes import BSParameters, OptionGreeks
from src.pricing.factory import PricingEngineFactory
from src.pricing.implied_vol import implied_volatility
from src.pricing.exotic import (
    ExoticParameters,
    BarrierType,
    AsianType,
    StrikeType,
    price_exotic_option
)
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams

from src.utils.circuit_breaker import pricing_circuit

import structlog

router = APIRouter(prefix="/pricing", tags=["pricing"])
logger = structlog.get_logger()


def _get_bs_params(request: Union[PriceRequest, GreeksRequest]) -> BSParameters:
    """Helper to convert API request to internal BSParameters."""
    return BSParameters(
        spot=request.spot,
        strike=request.strike,
        maturity=request.time_to_expiry,
        volatility=request.volatility,
        rate=request.rate,
        dividend=request.dividend_yield,
    )


@router.post(
    "/price",
    response_model=DataResponse[PriceResponse],
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@pricing_circuit
async def calculate_price(
    request: PriceRequest,
    response: Response,
    user: Any = Depends(get_current_user_flexible),
    redis_client: Any = Depends(get_redis_client),
    _: Any = Depends(rate_limit)
):
    """
    Calculate the theoretical price of a single option.

    Supports multiple models:
    - **black_scholes**: Standard analytical model
    - **monte_carlo**: Stochastic simulation
    - **binomial**: Cox-Ross-Rubinstein lattice model
    - **heston**: Stochastic volatility model (requires 'symbol')
    """
    start_time = time.perf_counter()
    params = _get_bs_params(request)

    # 1. Heston Logic
    if request.model == "heston":
        if not request.symbol:
            raise ValidationException(message="Symbol is required for Heston model")
        
        cache_key = f"heston_params:{request.symbol}"
        cached_data = await redis_client.get(cache_key)
        
        if cached_data:
            try:
                cache_obj = json.loads(cached_data)
                
                # Check for staleness (parameters older than 10 minutes are considered stale)
                params_age = time.time() - cache_obj.get("timestamp", 0)
                if params_age > 600:
                    logger.warning(
                        "heston_params_stale_falling_back",
                        symbol=request.symbol,
                        age_seconds=params_age
                    )
                else:
                    h_params = HestonParams(**cache_obj["params"])
                    model = HestonModelFFT(h_params, r=request.rate, T=request.time_to_expiry)
                    
                    if request.option_type == "call":
                        price = model.price_call(request.spot, request.strike)
                    else:
                        price = model.price_put(request.spot, request.strike)
                    
                    response.headers["X-Pricing-Model"] = "Heston-FFT"
                    
                    return DataResponse(
                        data=PriceResponse(
                            price=price,
                            spot=request.spot,
                            strike=request.strike,
                            time_to_expiry=request.time_to_expiry,
                            rate=request.rate,
                            volatility=request.volatility,
                            option_type=request.option_type,
                            model="heston",
                            cached=False,
                            computation_time_ms=(time.perf_counter() - start_time) * 1000,
                        )
                    )
            except Exception as e:
                logger.warning(
                    "heston_pricing_failed_falling_back",
                    symbol=request.symbol,
                    error=str(e)
                )
        
        # Fallback to Black-Scholes
        request.model = "black_scholes"
        response.headers["X-Pricing-Model"] = "Black-Scholes-Fallback"

    # 2. Standard Pricing Logic (Cache Check)
    cached_price = await pricing_cache.get_option_price(params, request.option_type, request.model)
    if cached_price is not None:
        return DataResponse(
            data=PriceResponse(
                price=cached_price,
                spot=request.spot,
                strike=request.strike,
                time_to_expiry=request.time_to_expiry,
                rate=request.rate,
                volatility=request.volatility,
                option_type=request.option_type,
                model=request.model,
                cached=True,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
            ),
            message="Result retrieved from cache"
        )

    # 3. Standard Pricing Logic (Execution)
    try:
        strategy = PricingEngineFactory.get_strategy(request.model)
        price = strategy.price(params, request.option_type)
        
        # Save to cache
        await pricing_cache.set_option_price(params, request.option_type, request.model, price)
        
    except ValueError as e:
        raise ValidationException(message=f"Invalid parameters for {request.model}: {str(e)}")
    except Exception as e:
        if "Circuit Breaker" in str(e):
            raise ServiceUnavailableException(message=str(e))
        logger.error(f"Unexpected error calculating price: {e}")
        raise InternalServerException(message="Internal error during price calculation")

    computation_time = (time.perf_counter() - start_time) * 1000

    return DataResponse(
        data=PriceResponse(
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
    )


@router.post(
    "/batch",
    response_model=DataResponse[BatchPriceResponse],
    responses={400: {"model": ErrorResponse}},
)
async def calculate_batch_prices(request: BatchPriceRequest):
    """
    Calculate prices for multiple options in a single batch request.

    Useful for portfolio-wide valuations. Maximum 1000 options per request.
    Individual failures in the batch are skipped and logged.
    """
    start_time = time.perf_counter()
    results = []
    cached_count = 0

    for opt_req in request.options:
        try:
            params = _get_bs_params(opt_req)
            
            # Try cache
            price = await pricing_cache.get_option_price(params, opt_req.option_type, opt_req.model)
            if price is not None:
                cached_count += 1
            else:
                strategy = PricingEngineFactory.get_strategy(opt_req.model)
                price = strategy.price(params, opt_req.option_type)
                await pricing_cache.set_option_price(params, opt_req.option_type, opt_req.model, price)

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
                    computation_time_ms=0.0,
                )
            )
        except Exception as e:
            logger.warning(f"Error in batch calculation for option {opt_req}: {e}")
            continue

    computation_time = (time.perf_counter() - start_time) * 1000

    return DataResponse(
        data=BatchPriceResponse(
            results=results,
            total_count=len(results),
            computation_time_ms=computation_time,
            cached_count=cached_count,
        )
    )


@router.post(
    "/greeks",
    response_model=DataResponse[GreeksResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
@pricing_circuit
async def calculate_greeks(
    request: GreeksRequest,
    user: Any = Depends(get_current_user_flexible),
    _: Any = Depends(rate_limit)
):
    """
    Calculate all standard option Greeks (Delta, Gamma, Theta, Vega, Rho).

    Currently uses the Black-Scholes analytical model for Greeks.
    """
    params = _get_bs_params(request)

    # Try cache
    cached_greeks = await pricing_cache.get_greeks(params, request.option_type)
    
    try:
        strategy = PricingEngineFactory.get_strategy("black_scholes")
        
        if cached_greeks:
            greeks = cached_greeks
        else:
            greeks_res = strategy.calculate_greeks(params, request.option_type)
            greeks = cast(OptionGreeks, greeks_res)
            await pricing_cache.set_greeks(params, request.option_type, greeks)
            
        price = strategy.price(params, request.option_type)
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        raise ValidationException(message=f"Greeks calculation failed: {str(e)}")

    return DataResponse(
        data=GreeksResponse(
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
    )


@router.post(
    "/implied-volatility",
    response_model=DataResponse[ImpliedVolatilityResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def calculate_iv(request: ImpliedVolatilityRequest):
    """
    Calculate the implied volatility from a given market price.

    Uses the Newton-Raphson numerical method.
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
    except ValueError as e:
        raise ValidationException(message=f"IV calculation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in IV calculation: {e}")
        raise InternalServerException(message="Internal error during IV calculation")

    return DataResponse(
        data=ImpliedVolatilityResponse(
            implied_volatility=iv,
            option_price=request.option_price,
            spot=request.spot,
            strike=request.strike,
            iterations=0,
            converged=True,
        )
    )


@router.post(
    "/exotic",
    response_model=DataResponse[ExoticPriceResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def calculate_exotic_price(request: ExoticPriceRequest):
    """
    Price exotic options:
    - **asian**: Geometric (analytical) or Arithmetic (Monte Carlo)
    - **barrier**: Knock-in/out (analytical)
    - **lookback**: Floating/Fixed strike (analytical/MC)
    - **digital**: Cash-or-nothing (analytical)
    """
    base_params = BSParameters(
        spot=request.spot,
        strike=request.strike,
        maturity=request.time_to_expiry,
        volatility=request.volatility,
        rate=request.rate,
        dividend=request.dividend_yield,
    )
    
    exotic_params = ExoticParameters(
        base_params=base_params,
        n_observations=request.n_observations,
        barrier=request.barrier or 0.0,
        rebate=request.rebate or 0.0
    )

    kwargs: Dict[str, Any] = {}
    if request.exotic_type == "barrier":
        if not request.barrier_type:
            raise ValidationException(message="barrier_type required for barrier options")
        try:
            kwargs["barrier_type"] = BarrierType[request.barrier_type.upper().replace("-", "_")]
        except KeyError:
            raise ValidationException(message=f"Invalid barrier_type: {request.barrier_type}")
            
    if request.exotic_type == "asian":
        kwargs["asian_type"] = AsianType[request.asian_type.upper()] if request.asian_type else AsianType.GEOMETRIC
        
    if request.exotic_type in ["asian", "lookback"]:
        kwargs["strike_type"] = StrikeType[request.strike_type.upper()] if request.strike_type else StrikeType.FIXED

    if request.exotic_type == "digital":
        kwargs["payout"] = request.payout

    try:
        price, ci = price_exotic_option(
            request.exotic_type, 
            exotic_params, 
            request.option_type, 
            **kwargs
        )
    except Exception as e:
        logger.error(f"Exotic pricing error: {e}")
        raise ValidationException(message=f"Pricing failed: {str(e)}")

    return DataResponse(
        data=ExoticPriceResponse(
            price=price,
            confidence_interval=ci,
            exotic_type=request.exotic_type
        )
    )