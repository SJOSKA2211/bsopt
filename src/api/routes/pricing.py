"""
Pricing Routes
==============

RESTful API endpoints for options pricing and Greeks calculation.
"""

import time
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
from src.pricing.black_scholes import BSParameters
from src.pricing.factory import PricingEngineFactory
from src.pricing.implied_vol import implied_volatility
from src.pricing.exotic import (
    ExoticParameters,
    BarrierType,
    AsianType,
    StrikeType,
    price_exotic_option
)
from src.utils.cache import get_redis_client, pricing_cache
from src.services.pricing_service import PricingService
from src.utils.circuit_breaker import pricing_circuit

import structlog

router = APIRouter(prefix="/pricing", tags=["pricing"])
logger = structlog.get_logger()
pricing_service = PricingService()


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
    """
    params = pricing_service.map_request_to_params(request)

    try:
        result = await pricing_service.price_option(
            params=params,
            option_type=request.option_type,
            model=request.model,
            symbol=request.symbol,
            redis_client=redis_client
        )
        
        response.headers["X-Pricing-Model"] = result.model
        
        return DataResponse(data=result)
        
    except ValueError as e:
        raise ValidationException(message=f"Invalid parameters: {str(e)}")
    except Exception as e:
        if "Circuit Breaker" in str(e):
            raise ServiceUnavailableException(message=str(e))
        logger.error("pricing_route_failed", error=str(e), exc_info=True)
        raise InternalServerException(message=f"Internal error: {str(e)}")


@router.post(
    "/batch",
    response_model=DataResponse[BatchPriceResponse],
    responses={400: {"model": ErrorResponse}},
)
async def calculate_batch_prices(request: BatchPriceRequest):
    """
    Calculate prices for multiple options in a single batch request.
    Optimized for high-throughput vectorized execution.
    """
    try:
        result = await pricing_service.price_batch(request.options)
        
        return DataResponse(data=result)
    except Exception as e:
        logger.error("batch_route_failed", error=str(e))
        raise InternalServerException(message="Batch calculation failed")


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
    Calculate all standard option Greeks.
    """
    params = pricing_service.map_request_to_params(request)

    try:
        result = await pricing_service.calculate_greeks(params, request.option_type)
        
        return DataResponse(
            data=GreeksResponse(
                delta=result["delta"],
                gamma=result["gamma"],
                theta=result["theta"],
                vega=result["vega"],
                rho=result["rho"],
                option_price=result["price"],
                spot=request.spot,
                strike=request.strike,
                time_to_expiry=request.time_to_expiry,
                volatility=request.volatility,
                option_type=request.option_type,
            )
        )
    except Exception as e:
        logger.error("greeks_route_failed", error=str(e))
        raise ValidationException(message=f"Greeks calculation failed: {str(e)}")


@router.post(
    "/implied-volatility",
    response_model=DataResponse[ImpliedVolatilityResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def calculate_iv(request: ImpliedVolatilityRequest):
    """
    Calculate the implied volatility.
    """
    try:
        iv = await pricing_service.calculate_iv(
            market_price=request.option_price,
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            rate=request.rate,
            dividend=request.dividend_yield,
            option_type=request.option_type,
        )
        
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
    except ValueError as e:
        raise ValidationException(message=f"IV calculation failed: {str(e)}")
    except Exception as e:
        logger.error("iv_route_failed", error=str(e), exc_info=True)
        raise InternalServerException(message="Internal error during IV calculation")


@router.post(
    "/exotic",
    response_model=DataResponse[ExoticPriceResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def calculate_exotic_price(request: ExoticPriceRequest):
    """
    Price exotic options.
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
            raise ValidationException(message="barrier_type required")
        try:
            kwargs["barrier_type"] = BarrierType[request.barrier_type.upper().replace("-", "_")]
        except KeyError:
            raise ValidationException(message=f"Invalid barrier type: {request.barrier_type}")
            
    if request.exotic_type == "asian":
        kwargs["asian_type"] = AsianType[request.asian_type.upper()] if request.asian_type else AsianType.GEOMETRIC
        
    if request.exotic_type in ["asian", "lookback"]:
        kwargs["strike_type"] = StrikeType[request.strike_type.upper()] if request.strike_type else StrikeType.FIXED

    if request.exotic_type == "digital":
        kwargs["payout"] = request.payout

    try:
        price, ci = await pricing_service.price_exotic(
            request.exotic_type, 
            exotic_params, 
            request.option_type, 
            **kwargs
        )
        
        return DataResponse(
            data=ExoticPriceResponse(
                price=price,
                confidence_interval=ci,
                exotic_type=request.exotic_type
            )
        )
    except Exception as e:
        logger.error("exotic_route_failed", error=str(e))
        raise ValidationException(message=f"Pricing failed: {str(e)}")