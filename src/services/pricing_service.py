"""
Pricing Service (Singularity Refactored)
"""

import time
from typing import Any

import structlog
from fastapi import HTTPException  # Added HTTPException

from src.api.schemas.pricing import BatchPriceResponse, PriceResponse
from src.pricing.black_scholes import BSParameters
from src.pricing.factory import (  # Added PricingEngineNotFound
    PricingEngineFactory,
    PricingEngineNotFound,
)

logger = structlog.get_logger(__name__)


class PricingService:
    """
    Service for unified option pricing using vectorized strategies.
    """

    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: str | None = None,
    ) -> PriceResponse:
        start_time = time.perf_counter()

        try:
            engine = PricingEngineFactory.get_engine(model)
        except PricingEngineNotFound as e:
            logger.error("pricing_engine_not_found", model=model, error=str(e))
            raise HTTPException(
                status_code=400, detail=f"Invalid pricing model '{model}': {str(e)}"
            )
        except Exception as e:
            logger.error("pricing_engine_factory_error", model=model, error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Error initializing pricing engine: {str(e)}"
            )

        from anyio.to_thread import run_sync

        try:
            price = await run_sync(engine.price, params, option_type)
        except Exception as e:
            logger.error(
                "pricing_engine_calculation_error",
                model=model,
                error=str(e),
                params=params.dict(),
            )
            raise HTTPException(
                status_code=422, detail=f"Pricing calculation failed: {str(e)}"
            )

        return PriceResponse.model_construct(
            price=price,
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.time_to_expiry,
            rate=params.rate,
            volatility=params.volatility,
            option_type=option_type,
            model=model,
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def price_batch(self, options: list[Any]) -> BatchPriceResponse:
        """
        🚀 OPTIMIZATION: True Batch Pricing.
        Groups options by model and uses vectorized engines for massive performance gains.
        """
        start_time = time.perf_counter()

        # 1. Group options by model
        from collections import defaultdict

        model_groups = defaultdict(list)
        for o in options:
            model_groups[o.model].append(o)

        results = [None] * len(options)
        option_to_idx = {id(o): i for i, o in enumerate(options)}

        # 2. Process each group vectorially
        for model, group in model_groups.items():
            try:
                engine = PricingEngineFactory.get_engine(model)

                # Extract parameters for vectorization
                spots = np.array([o.spot for o in group], dtype=np.float64)
                strikes = np.array([o.strike for o in group], dtype=np.float64)
                maturities = np.array(
                    [o.time_to_expiry for o in group], dtype=np.float64
                )
                vols = np.array([o.volatility for o in group], dtype=np.float64)
                rates = np.array([o.rate for o in group], dtype=np.float64)
                types = np.array([o.option_type for o in group])

                # Vectorized call
                # Offload to thread pool for heavy math
                from anyio.to_thread import run_sync

                prices = await run_sync(
                    engine.price_options,
                    spots,
                    strikes,
                    maturities,
                    vols,
                    rates,
                    0.0,
                    types,
                )

                # Reconstruct responses
                for i, o in enumerate(group):
                    idx = option_to_idx[id(o)]
                    results[idx] = PriceResponse.model_construct(
                        price=(
                            float(prices[i])
                            if isinstance(prices, np.ndarray)
                            else float(prices)
                        ),
                        spot=o.spot,
                        strike=o.strike,
                        time_to_expiry=o.time_to_expiry,
                        rate=o.rate,
                        volatility=o.volatility,
                        option_type=o.option_type,
                        model=model,
                        computation_time_ms=0,  # Grouped time is tracked at batch level
                    )
            except Exception as e:
                logger.error("vectorized_batch_failed", model=model, error=str(e))
                # Fallback to individual error reporting for the group
                for o in group:
                    idx = option_to_idx[id(o)]
                    results[idx] = PriceResponse.model_construct(
                        error=f"Batch processing failed for {model}: {str(e)}",
                        spot=o.spot,
                        strike=o.strike,
                        time_to_expiry=o.time_to_expiry,
                        rate=o.rate,
                        volatility=o.volatility,
                        option_type=o.option_type,
                        model=model,
                        computation_time_ms=0,
                    )

        return BatchPriceResponse(
            results=results,
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def calculate_greeks(self, params: BSParameters, option_type: str) -> dict:
        engine = PricingEngineFactory.get_engine("black_scholes")
        from anyio.to_thread import run_sync

        greeks = await run_sync(engine.calculate_greeks, params, option_type)
        return greeks.__dict__
