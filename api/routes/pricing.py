"""
Pricing Routes (Optimized)
"""

import datetime

import msgspec
import structlog
from fastapi import APIRouter, Depends, Request

from api.responses import MsgspecJSONResponse
from api.schemas.pricing import (
    BatchGreeksRequest,
    BatchGreeksResult,
    BatchPriceRequest,
    BatchPriceResult,
    BatchPriceResult,
    GreeksRequest,
    HeatmapRequest,
    PriceRequest,
    PriceResult,
)
from src.auth.auth import get_current_active_user
from src.database.models import User
from src.math_kernel.service import PricingService
from src.shared.utils.cache import multi_layer_cache
from src.shared.utils.circuit_breaker import pricing_circuit

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["Pricing"], default_response_class=MsgspecJSONResponse)
pricing_service = PricingService()


@router.post("/price", response_model=None)
@pricing_circuit
@multi_layer_cache(prefix="price", ttl=300)
async def calculate_price(
    body: PriceRequest, request: Request, current_user: User = Depends(get_current_active_user)
) -> MsgspecJSONResponse:
    """
    Calculate theoretical price for a single option.
    OPTIMIZED: Returns msgspec Struct for ultra-fast serialization.
    """
    params = body.to_bs_params()
    res = await pricing_service.price_option(
        params=params,
        option_type=body.option_type,
        model=body.model,
        symbol=body.symbol,
    )
    return MsgspecJSONResponse(content=res)


@router.post("/batch", response_model=None)
@pricing_circuit
@multi_layer_cache(prefix="batch_price", ttl=60)
async def calculate_batch_prices(
    request: BatchPriceRequest, current_user: User = Depends(get_current_active_user)
) -> MsgspecJSONResponse:
    """
    Vectorized batch pricing.
    OPTIMIZED: Zero-overhead batch response using msgspec Structs.
    """
    res = await pricing_service.price_batch(request.options)
    return MsgspecJSONResponse(content=res)


@router.post("/greeks/batch", response_model=None)
@pricing_circuit
async def calculate_batch_greeks(
    request: BatchGreeksRequest, current_user: User = Depends(get_current_active_user)
) -> MsgspecJSONResponse:
    """
    Vectorized batch Greek calculation.
    """
    res = await pricing_service.calculate_greeks_batch(request.options)
    return MsgspecJSONResponse(content=res)


@router.post("/greeks", response_class=MsgspecJSONResponse)
@pricing_circuit
@multi_layer_cache(prefix="greeks", ttl=300)
async def calculate_greeks(
    body: GreeksRequest, current_user: User = Depends(get_current_active_user)
):
    """
    Calculate option Greeks.
    """
    params = body.to_bs_params()
    result = await pricing_service.calculate_greeks(params, body.option_type)
    return MsgspecJSONResponse(content=result)


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
@pricing_circuit
async def calculate(
    body: dict, current_user: User = Depends(get_current_active_user)
) -> MsgspecJSONResponse:
    """
    Convenience endpoint used by tests and demos.
    Maps {s, k, t, r, sigma, option_type, model} → {price, greeks}.
    OPTIMIZED: Returns high-performance MsgspecJSONResponse.
    """
    from api.schemas.pricing import PriceRequest as PR  # noqa: N817

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

    # Calculate price and implicitly yield greeks natively
    try:
        result = await pricing_service.price_option(
            params=params,
            option_type=req.option_type,
            model=req.model,
            symbol=req.symbol,
        )
    except Exception as e:
        logger.error("calculate_price_failed", error=str(e))
        result = None

    greeks_data = {}
    if not result:
        return _build_calculate_response(None, req, greeks_data)

    greeks_raw = getattr(result, "greeks", None)
    if greeks_raw:
        if hasattr(greeks_raw, "__dict__"):
            greeks_data = vars(greeks_raw)
        elif isinstance(greeks_raw, dict):
            greeks_data = greeks_raw
        greeks_data = {k: float(v) for k, v in greeks_data.items() if v is not None}

    return _build_calculate_response(result, req, greeks_data)


@router.post("/heatmap")
@pricing_circuit
async def calculate_heatmap(
    request: HeatmapRequest, current_user: User = Depends(get_current_active_user)
) -> MsgspecJSONResponse:
    """
    Generate multidimensional risk heatmap.
    """
    res = await pricing_service.generate_heatmap(request)
    return MsgspecJSONResponse(content=res)


def _build_calculate_response(result, req, greeks_data) -> MsgspecJSONResponse:
    resp = CalculateResponseStruct(
        price=getattr(result, "price", 0.0) if result else 0.0,
        greeks=greeks_data,
        spot=getattr(result, "spot", req.spot) if result else req.spot,
        strike=getattr(result, "strike", req.strike) if result else req.strike,
        time_to_expiry=getattr(result, "time_to_expiry", req.time_to_expiry)
        if result
        else req.time_to_expiry,
        rate=getattr(result, "rate", req.rate) if result else req.rate,
        volatility=getattr(result, "volatility", req.volatility) if result else req.volatility,
        option_type=getattr(result, "option_type", req.option_type) if result else req.option_type,
        model=getattr(result, "model", req.model) if result else req.model,
        computation_time_ms=getattr(result, "computation_time_ms", 0.0) if result else 0.0,
        cached=getattr(result, "cached", False) if result else False,
        timestamp=getattr(result, "timestamp", datetime.datetime.now(datetime.UTC)),
    )
    return MsgspecJSONResponse(content=resp)