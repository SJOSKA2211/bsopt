"""
Pricing Service (Singularity Refactored)
"""

import time
import asyncio
import structlog
from typing import Any, List, Optional
from src.pricing.factory import PricingEngineFactory
from src.api.schemas.pricing import PriceResponse, BatchPriceResponse
from src.pricing.black_scholes import BSParameters
from src.config import settings

logger = structlog.get_logger(__name__)

class PricingService:
    """
    Unified Service for Option Pricing.
    Uses PricingEngineFactory for strategy selection.
    """

    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: Optional[str] = None
    ) -> PriceResponse:
        start_time = time.perf_counter()
        
        # Get optimal engine from factory
        engine = PricingEngineFactory.get_engine(model)
        
        # Offload math to thread pool to keep event loop free
        from anyio.to_thread import run_sync
        price = await run_sync(engine.price, params, option_type)
        
        return PriceResponse.model_construct(
            price=price,
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.maturity,
            rate=params.rate,
            volatility=params.volatility,
            option_type=option_type,
            model=model,
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )

    async def price_batch(self, options: List[Any]) -> BatchPriceResponse:
        start_time = time.perf_counter()
        
        # For now, process concurrently (we'll add vectorized array support back later)
        tasks = [
            self.price_option(
                o.to_bs_params(), o.option_type, o.model, o.symbol
            ) for o in options
        ]
        results = await asyncio.gather(*tasks)
        
        return BatchPriceResponse(
            results=results,
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )

    async def calculate_greeks(self, params: BSParameters, option_type: str) -> dict:
        engine = PricingEngineFactory.get_engine("black_scholes")
        from anyio.to_thread import run_sync
        greeks = await run_sync(engine.calculate_greeks, params, option_type)
        return greeks.__dict__