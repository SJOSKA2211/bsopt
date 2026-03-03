"""
Pricing Routes (Optimized Refactored)
"""

import structlog
from fastapi import APIRouter, Request

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.pricing import BatchPriceRequest, GreeksRequest, PriceRequest
from src.services.pricing_service import PricingService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/pricing", tags=["Pricing"])
pricing_service = PricingService()


@router.post("/price", response_class=MsgspecJSONResponse)
async def calculate_price(body: PriceRequest, request: Request):
    """
    Calculate theoretical price for a single option.
    """
    params = body.to_bs_params()
    result = await pricing_service.price_option(
        params=params,
        option_type=body.option_type,
        model=body.model,
        symbol=body.symbol,
    )
    return result


@router.post("/batch", response_class=MsgspecJSONResponse)
async def calculate_batch_prices(request: BatchPriceRequest):
    """
    Vectorized batch pricing.
    """
    result = await pricing_service.price_batch(request.options)
    return result


@router.post("/greeks", response_class=MsgspecJSONResponse)
async def calculate_greeks(body: GreeksRequest):
    """
    Calculate option Greeks.
    """
    params = body.to_bs_params()
    result = await pricing_service.calculate_greeks(params, body.option_type)
    return result


@router.post("/calculate")
async def calculate(body: dict) -> dict:
    """
    Convenience endpoint used by tests and demos.
    Maps {s, k, t, r, sigma, option_type, model} → {price, greeks}.
    """
    from src.api.schemas.pricing import PriceRequest as PR

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

    # Convert price result struct to plain dict safely
    try:
        import msgspec
        data = msgspec.to_builtins(result)
    except Exception:
        try:
            data = result.model_dump() if hasattr(result, "model_dump") else vars(result)
        except Exception:
            data = {"price": 0.0}

    price = data.get("price") or 0.0
    return {"price": price, "greeks": greeks_data, **data}



