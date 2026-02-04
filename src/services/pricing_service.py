"""
Pricing Service (Singularity Refactored)
"""

import time
import asyncio
import structlog
from typing import Any, List, Optional
from src.pricing.factory import PricingEngineFactory, PricingEngineNotFound # Added PricingEngineNotFound
from src.api.schemas.pricing import PriceResponse, BatchPriceResponse
from src.pricing.black_scholes import BSParameters
from src.config import settings
from fastapi import HTTPException # Added HTTPException

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
        
        try:
            # Get optimal engine from factory
            engine = PricingEngineFactory.get_engine(model)
        except PricingEngineNotFound as e: # Catch specific factory errors
            logger.error("pricing_engine_not_found", model=model, error=str(e))
            raise HTTPException(status_code=400, detail=f"Invalid pricing model '{model}': {str(e)}")
        except Exception as e:
            logger.error("pricing_engine_factory_error", model=model, error=str(e))
            raise HTTPException(status_code=500, detail=f"Error initializing pricing engine: {str(e)}")
        
        # Offload math to thread pool to keep event loop free
        from anyio.to_thread import run_sync
        try:
            price = await run_sync(engine.price, params, option_type)
        except Exception as e: # Catch exceptions from the synchronous pricing engine
            logger.error("pricing_engine_calculation_error", model=model, error=str(e), params=params.dict())
            raise HTTPException(status_code=422, detail=f"Pricing calculation failed: {str(e)}")
        
        return PriceResponse.model_construct(
            price=price,
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.time_to_expiry, # Fixed typo here!
            rate=params.rate,
            volatility=params.volatility,
            option_type=option_type,
            model=model,
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )

    async def price_batch(self, options: List[Any]) -> BatchPriceResponse:
        start_time = time.perf_counter()
        
        tasks = []
        # Temporarily store original request data for better error reporting
        option_requests = [] 
        for o in options:
            try:
                tasks.append(
                    self.price_option(
                        o.to_bs_params(), o.option_type, o.model, o.symbol
                    )
                )
                option_requests.append(o) # Store valid requests
            except Exception as e:
                # If to_bs_params fails, create an immediate error result
                tasks.append(asyncio.sleep(0, result={"error": f"Invalid input for option: {str(e)}"}))
                option_requests.append(o) # Store even invalid ones for context

        # Use return_exceptions=True to allow individual task failures without stopping the whole batch
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                # Convert exceptions to a consistent error format
                logger.error("batch_option_failed", error=str(res), option=option_requests[i].dict())
                results.append(PriceResponse.model_construct(
                    error=f"Processing failed: {str(res)}", 
                    spot=option_requests[i].spot,
                    strike=option_requests[i].strike,
                    time_to_expiry=option_requests[i].time_to_expiry,
                    rate=option_requests[i].rate,
                    volatility=option_requests[i].volatility,
                    option_type=option_requests[i].option_type,
                    model=option_requests[i].model,
                    computation_time_ms=0
                ))
            elif isinstance(res, dict) and "error" in res:
                # Handle the immediate error results from to_bs_params failure
                results.append(PriceResponse.model_construct(
                    error=res["error"],
                    spot=option_requests[i].spot,
                    strike=option_requests[i].strike,
                    time_to_expiry=option_requests[i].time_to_expiry,
                    rate=option_requests[i].rate,
                    volatility=option_requests[i].volatility,
                    option_type=option_requests[i].option_type,
                    model=option_requests[i].model,
                    computation_time_ms=0
                ))
            else:
                results.append(res)
        
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