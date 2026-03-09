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

    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: str = None,
    ) -> PriceResult:
        """
        Calculates price and greeks for a single option request.
        Dispatches to the optimized engine determined by the model type.
        """
        try:
            # 1. Resolve Engine
            engine = self.factory.get_engine(model)

            # 2. Compute (Off-load to thread pool for heavy engines)
            start_time = time.perf_counter()
            result = await run_sync(engine.price_european, params)
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "option_priced",
                model=model,
                duration_ms=round(duration_ms, 2),
            )

            from src.api.schemas.pricing import OptionGreeksStruct

            return PriceResult(
                price=result.price,
                spot=params.spot,
                strike=params.strike,
                time_to_expiry=params.maturity,
                rate=params.rate,
                volatility=params.volatility,
                option_type=option_type,
                model=model,
                computation_time_ms=duration_ms,
                greeks=OptionGreeksStruct(
                    delta=result.greeks.delta,
                    gamma=result.greeks.gamma,
                    theta=result.greeks.theta,
                    vega=result.greeks.vega,
                    rho=result.greeks.rho,
                )
                if result.greeks
                else None,
            )

        except PricingEngineNotFound as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("pricing_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal pricing error")

    async def price_batch(self, request: Any) -> BatchPriceResult:
        """
        HIGH-PERFORMANCE: Prices an array of options concurrently using vectorized group-batching.
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

                # Check for vectorized capability (High-Performance check)
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

                    from src.api.schemas.pricing import OptionGreeksStruct

                    for (original_idx, _), res in zip(items, batch_results):
                        results[original_idx] = PriceResult(
                            price=res.price,
                            spot=res.spot,
                            strike=res.strike,
                            time_to_expiry=res.maturity,
                            rate=res.rate,
                            volatility=res.volatility,
                            option_type=res.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=res.greeks.delta,
                                gamma=res.greeks.gamma,
                                theta=res.greeks.theta,
                                vega=res.greeks.vega,
                                rho=res.greeks.rho,
                            )
                            if res.greeks
                            else None,
                        )
                else:
                    # Fallback to concurrent scalar pricing
                    from src.api.schemas.pricing import OptionGreeksStruct

                    for original_idx, item_req in items:
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
                            spot=item_req.spot,
                            strike=item_req.strike,
                            time_to_expiry=item_req.maturity,
                            rate=item_req.rate,
                            volatility=item_req.volatility,
                            option_type=item_req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=res.greeks.delta,
                                gamma=res.greeks.gamma,
                                theta=res.greeks.theta,
                                vega=res.greeks.vega,
                                rho=res.greeks.rho,
                            )
                            if res.greeks
                            else None,
                        )

            except Exception as exc:
                logger.error("group_pricing_failed", model=model_type, error=str(exc))
                # Fill gaps with error indicator or zero
                for original_idx, item_req in items:
                    results[original_idx] = PriceResult(
                        price=0.0,
                        spot=item_req.spot,
                        strike=item_req.strike,
                        time_to_expiry=item_req.maturity,
                        rate=item_req.rate,
                        volatility=item_req.volatility,
                        option_type=item_req.option_type,
                        model=model_type,
                        computation_time_ms=0.0,
                        greeks=None,
                    )

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
