"""
Unified Pricing Service
"""

import time
from collections import defaultdict
from typing import Any

import numpy as np
import structlog
from anyio.to_thread import run_sync
from fastapi import HTTPException

from src.api.schemas.pricing import BatchPriceResponse, PriceResponse
from src.pricing.black_scholes import BSParameters
from src.pricing.factory import PricingEngineFactory, PricingEngineNotFound

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
            ) from e

        try:
            price = await run_sync(engine.price, params, option_type)
        except Exception as e:
            logger.error("pricing_engine_calculation_error", model=model, error=str(e))
            raise HTTPException(
                status_code=422, detail=f"Pricing calculation failed: {str(e)}"
            ) from e

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
        Efficient Batch Pricing using vectorized engines.
        OPTIMIZED: Model grouping and JIT-accelerated vectorized calculation.
        """
        start_time = time.perf_counter()

        # 1. Group options by model with their original indices
        model_groups = defaultdict(list)
        for i, o in enumerate(options):
            model_groups[o.model].append((i, o))

        results = [None] * len(options)

        # 2. Process each model group
        for model, group in model_groups.items():
            try:
                engine = PricingEngineFactory.get_engine(model)
                [item[0] for item in group]
                items = [item[1] for item in group]

                # Extract parameters for vectorization
                spots = np.array([o.spot for o in items], dtype=np.float64)
                strikes = np.array([o.strike for o in items], dtype=np.float64)
                maturities = np.array(
                    [o.time_to_expiry for o in items], dtype=np.float64
                )
                vols = np.array([o.volatility for o in items], dtype=np.float64)
                rates = np.array([o.rate for o in items], dtype=np.float64)
                types = np.array([o.option_type for o in items])

                # JIT-accelerated vectorized call
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

                # Map results back using the stored indices
                for k, (orig_idx, o) in enumerate(group):
                    results[orig_idx] = PriceResponse.model_construct(
                        price=(
                            float(prices[k])
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
                        computation_time_ms=0,  # Included in total
                    )
            except Exception as e:
                logger.error("batch_group_processing_failed", model=model, error=str(e))
                for orig_idx, o in group:
                    results[orig_idx] = PriceResponse.model_construct(
                        error=f"Batch failed: {str(e)}",
                        spot=o.spot,
                        strike=o.strike,
                        time_to_expiry=o.time_to_expiry,
                        model=model,
                    )

        return BatchPriceResponse(
            results=results,
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def calculate_greeks(self, params: BSParameters, option_type: str) -> dict:
        engine = PricingEngineFactory.get_engine("black_scholes")
        greeks = await run_sync(engine.calculate_greeks, params, option_type)
        return greeks.__dict__

    def clear_cache(self):
        """Mock for test compatibility."""
        pass
