"""
Unified Pricing Service
"""

import asyncio
import time
from collections import defaultdict
from typing import Any, cast

import numpy as np
import structlog
from anyio.to_thread import run_sync
from fastapi import HTTPException

from src.api.schemas.pricing import (
    BatchPriceResult,
    PriceResult,
)
from src.pricing.black_scholes import BSParameters
from src.pricing.factory import PricingEngineFactory, PricingEngineNotFound

logger = structlog.get_logger(__name__)


class PricingService:
    """
    Unified entry point for all option pricing operations.
    Supports single and batch pricing using high-performance engines.
    """

    def __init__(self, factory: PricingEngineFactory = None):
        self.factory = factory or PricingEngineFactory()

    async def price_option(self, request: Any) -> PriceResult:
        """
        Calculates price and greeks for a single option request.
        Dispatches to the optimized engine determined by the model type.
        """
        try:
            # 1. Resolve Engine
            engine = self.factory.get_engine(request.model_type)

            # 2. Extract Parameters
            params = BSParameters(
                spot=request.spot,
                strike=request.strike,
                maturity=request.maturity,
                volatility=request.volatility,
                rate=request.rate,
                dividend=request.dividend,
            )

            # 3. Compute (Off-load to thread pool for heavy engines)
            start_time = time.perf_counter()
            result = await run_sync(engine.price_european, params)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "option_priced",
                model=request.model_type,
                duration_ms=round(duration_ms, 2),
            )

            return PriceResult(
                price=result.price,
                greeks=result.greeks.__dict__,
                computation_time_ms=duration_ms,
            )

        except PricingEngineNotFound as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("pricing_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal pricing error")

    async def price_batch(self, request: Any) -> BatchPriceResult:
        """
        GOD-MODE: Prices an array of options concurrently using vectorized group-batching.
        Groups requests by model type to maximize SIMD efficiency.
        """
        start_time = time.perf_counter()
        results: list[PriceResult | None] = [None] * len(request.requests)

        # 1. Group by model type
        model_groups = defaultdict(list)
        for i, req in enumerate(request.requests):
            model_groups[req.model_type].append((i, req))

        # 2. Process each group (Vectorized if engine supports it)
        async def _process_group(model_type: str, items: list[tuple[int, Any]]):
            try:
                engine = self.factory.get_engine(model_type)

                # Check for vectorized capability (God-Mode check)
                if hasattr(engine, "price_batch_vectorized"):
                    # Extract params into numpy arrays
                    spots = np.array([it[1].spot for it in items], dtype=np.float64)
                    strikes = np.array([it[1].strike for it in items], dtype=np.float64)
                    maturities = np.array([it[1].maturity for it in items], dtype=np.float64)
                    vols = np.array([it[1].volatility for it in items], dtype=np.float64)
                    rates = np.array([it[1].rate for it in items], dtype=np.float64)
                    divs = np.array([it[1].dividend for it in items], dtype=np.float64)

                    params_batch = (spots, strikes, maturities, vols, rates, divs)
                    batch_results = await run_sync(engine.price_batch_vectorized, params_batch)

                    for (original_idx, _), res in zip(items, batch_results):
                        results[original_idx] = PriceResult(
                            price=res.price,
                            greeks=res.greeks.__dict__,
                            computation_time_ms=0.0,  # Batch level tracking
                        )
                else:
                    # Fallback to concurrent scalar pricing
                    for original_idx, item_req in items:
                        # Re-use logic from single price_option (simplified for batch)
                        params = BSParameters(
                            spot=item_req.spot,
                            strike=item_req.strike,
                            maturity=item_req.maturity,
                            volatility=item_req.volatility,
                            rate=item_req.rate,
                            dividend=item_req.dividend,
                        )
                        res = await run_sync(engine.price_european, params)
                        results[original_idx] = PriceResult(
                            price=res.price,
                            greeks=res.greeks.__dict__,
                            computation_time_ms=0.0,
                        )

            except Exception as exc:
                logger.error("group_pricing_failed", model=model_type, error=str(exc))
                # Fill gaps with error indicator or zero
                for original_idx, _ in items:
                    results[original_idx] = PriceResult(price=0.0, greeks={}, error=str(exc))

        # 3. Dispatch all groups concurrently
        tasks = [_process_group(m, g) for m, g in model_groups.items()]
        if tasks:
            await asyncio.gather(*tasks)

        return BatchPriceResult(
            results=cast(list[PriceResult], results),
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )


# Global Singleton for injection
pricing_service = PricingService()
