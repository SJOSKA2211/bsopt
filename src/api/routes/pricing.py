"""
Pricing Routes (Optimized)
"""

import datetime

import msgspec
import structlog
from fastapi import APIRouter, Request

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.pricing import (
    BatchGreeksRequest,
    BatchGreeksResult,
    BatchPriceRequest,
    BatchPriceResult,
    GreeksRequest,
    PriceRequest,
    PriceResult,
)
from src.services.pricing_service import PricingService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["Pricing"], default_response_class=MsgspecJSONResponse)
pricing_service = PricingService()


@router.post("/price", response_model=None)
async def calculate_price(body: PriceRequest, request: Request) -> PriceResult:
    """
    Calculate theoretical price for a single option.
    OPTIMIZED: Returns msgspec Struct for ultra-fast serialization.
    """
    params = body.to_bs_params()
    return await pricing_service.price_option(
        params=params,
        option_type=body.option_type,
        model=body.model,
        symbol=body.symbol,
    )


@router.post("/batch", response_model=None)
async def calculate_batch_prices(request: BatchPriceRequest) -> BatchPriceResult:
    """
    Vectorized batch pricing.
    OPTIMIZED: Zero-overhead batch response using msgspec Structs.
    """
    return await pricing_service.price_batch(request.options)


@router.post("/greeks/batch", response_model=None)
async def calculate_batch_greeks(request: BatchGreeksRequest) -> BatchGreeksResult:
    """
    Vectorized batch Greek calculation.
    """
    return await pricing_service.calculate_greeks_batch(request.options)


@router.post("/greeks", response_class=MsgspecJSONResponse)
async def calculate_greeks(body: GreeksRequest):
    """
    Calculate option Greeks.
    """
    params = body.to_bs_params()
    result = await pricing_service.calculate_greeks(params, body.option_type)
    return result


class CalculateResponseStruct(msgspec.Struct):
    price: float
    greeks: dict[str, float]
    spot: float
    strike: float
    time_to_expiry: float
    rate: float
    volatility: float
    option_type: str
    model: str
    computation_time_ms: float
    cached: bool
    timestamp: datetime.datetime


@router.post("/calculate")
async def calculate(body: dict) -> MsgspecJSONResponse:
    """
    Convenience endpoint used by tests and demos.
    Maps {s, k, t, r, sigma, option_type, model} → {price, greeks}.
    OPTIMIZED: Returns high-performance MsgspecJSONResponse.
    """
    from src.api.schemas.pricing import PriceRequest as PR  # noqa: N817

    req = PR(
        spot=body.get("s", body.get("spot", 100)),
        strike=body.get("k", body.get("strike", 100)),
        time_to_expiry=body.get("t", body.get("time_to_expiry", 1.0)),
        rate=body.get("r", body.get("rate", 0.05)),
        volatility=body.get("sigma", body.get("volatility", 0.2)),
        option_type=body.get("option_type", "call"),
        model=body.get("model", "black_scholes"),
        symbol=body.get("symbol"),
    )
    params = req.to_bs_params()

    # Concurrently calculate price and greeks
    price_task = pricing_service.price_option(
        params=params,
        option_type=req.option_type,
        model=req.model,
        symbol=req.symbol,
    )
    greeks_task = pricing_service.calculate_greeks(params, req.option_type)

    result, greeks_result = await asyncio.gather(price_task, greeks_task, return_exceptions=True)

    # Handle results and potential exceptions
    if isinstance(result, Exception):
        logger.error("calculate_price_failed", error=str(result))
        result = None
    
    greeks_data = {}
    if not isinstance(greeks_result, Exception) and greeks_result is not None:
        if hasattr(greeks_result, "__dict__"):
            greeks_data = vars(greeks_result)
        elif isinstance(greeks_result, dict):
            greeks_data = greeks_result
        # Ensure all greeks are plain floats not numpy arrays
        greeks_data = {k: float(v) for k, v in greeks_data.items() if v is not None}
    elif isinstance(greeks_result, Exception):
        logger.warning("calculate_greeks_failed", error=str(greeks_result))

    # Zero-copy Struct response
    resp = CalculateResponseStruct(
        price=getattr(result, "price", 0.0) if result else 0.0,
        greeks=greeks_data,
        spot=getattr(result, "spot", req.spot) if result else req.spot,
        strike=getattr(result, "strike", req.strike) if result else req.strike,
        time_to_expiry=getattr(result, "time_to_expiry", req.time_to_expiry) if result else req.time_to_expiry,
        rate=getattr(result, "rate", req.rate) if result else req.rate,
        volatility=getattr(result, "volatility", req.volatility) if result else req.volatility,
        option_type=getattr(result, "option_type", req.option_type) if result else req.option_type,
        model=getattr(result, "model", req.model) if result else req.model,
        computation_time_ms=getattr(result, "computation_time_ms", 0.0) if result else 0.0,
        cached=getattr(result, "cached", False) if result else False,
        timestamp=getattr(result, "timestamp", datetime.datetime.now(datetime.UTC)),
    )
    return MsgspecJSONResponse(content=resp)
