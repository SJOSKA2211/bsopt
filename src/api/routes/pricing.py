"""
Pricing Routes (Singularity Refactored)
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
async def calculate_price(
    body: PriceRequest,
    request: Request
):
    """
    Calculate theoretical price for a single option.
    """
    params = body.to_bs_params()
    result = await pricing_service.price_option(
        params=params,
        option_type=body.option_type,
        model=body.model,
        symbol=body.symbol
    )
    return result

@router.post("/batch", response_class=MsgspecJSONResponse)
async def calculate_batch_prices(
    request: BatchPriceRequest
):
    """
    Vectorized batch pricing.
    """
    result = await pricing_service.price_batch(request.options)
    return result

@router.post("/greeks", response_class=MsgspecJSONResponse)
async def calculate_greeks(
    body: GreeksRequest
):
    """
    Calculate option Greeks.
    """
    params = body.to_bs_params()
    result = await pricing_service.calculate_greeks(params, body.option_type)
    return result