"""
Pricing Routes (Optimized Refactored)
"""

import structlog
from fastapi import APIRouter, Request

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.pricing import (
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


@router.post("/greeks", response_class=MsgspecJSONResponse)
async def calculate_greeks(body: GreeksRequest):
    """
    Calculate option Greeks.
    """
    params = body.to_bs_params()
    result = await pricing_service.calculate_greeks(params, body.option_type)
    return result


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

    # Calculate price
    result = await pricing_service.price_option(
        params=params,
        option_type=req.option_type,
        model=req.model,
        symbol=req.symbol,
    )

    # Also calculate greeks
    greeks_data = {}
    try:
        greeks_result = await pricing_service.calculate_greeks(params, req.option_type)
        if hasattr(greeks_result, "__dict__"):
            greeks_data = vars(greeks_result)
        elif isinstance(greeks_result, dict):
            greeks_data = greeks_result
        # Ensure all greeks are plain floats not numpy arrays
        greeks_data = {k: float(v) for k, v in greeks_data.items() if v is not None}
    except Exception:
        pass

    # Convert result struct to dict safely for response
    import msgspec
    data = msgspec.to_builtins(result)
    
    return MsgspecJSONResponse(
        content={"price": data.get("price", 0.0), "greeks": greeks_data, **data}
    )
