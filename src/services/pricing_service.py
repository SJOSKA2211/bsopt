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

    async def price_batch(self, options: list[Any]) -> BatchPriceResult:
        """
        HIGH-PERFORMANCE: Prices an array of options concurrently using vectorized group-batching.
        Groups requests by model type to maximize SIMD efficiency.
        """
        start_time = time.perf_counter()
        results: list[PriceResult | None] = [None] * len(options)

        # 1. Group by model type
        model_groups = defaultdict(list)
        for i, req in enumerate(options):
            model_groups[req.model].append((i, req))

        # 2. Process each group (Vectorized if engine supports it)
        async def _process_group(model_type: str, items: list[tuple[int, Any]]):
            try:
                engine = self.factory.get_engine(model_type)

                # Vectorized parameters
                spots = np.array([it[1].spot for it in items], dtype=np.float64)
                strikes = np.array([it[1].strike for it in items], dtype=np.float64)
                maturities = np.array([it[1].time_to_expiry for it in items], dtype=np.float64)
                vols = np.array([it[1].volatility for it in items], dtype=np.float64)
                rates = np.array([it[1].rate for it in items], dtype=np.float64)
                divs = np.array([it[1].dividend_yield for it in items], dtype=np.float64)
                types = np.array([it[1].option_type for it in items])

                from src.api.schemas.pricing import OptionGreeksStruct

                # Optimized Dispatch
                if model_type == "black_scholes":
                    from src.pricing.black_scholes import BlackScholesEngine
                    
                    # Direct call to JIT/Rust batch kernels
                    prices = await run_sync(
                        BlackScholesEngine.price_options,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    
                    # Calculate Greeks in batch too
                    g_res = await run_sync(
                        BlackScholesEngine.calculate_greeks,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    
                    for idx, (original_idx, req) in enumerate(items):
                        results[original_idx] = PriceResult(
                            price=float(prices[idx]),
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=float(g_res.delta[idx]),
                                gamma=float(g_res.gamma[idx]),
                                theta=float(g_res.theta[idx]),
                                vega=float(g_res.vega[idx]),
                                rho=float(g_res.rho[idx]),
                            )
                        )
                elif model_type == "neural":
                    # Neural engine is already vectorized via PyTorch
                    prices = await run_sync(
                        engine.price_batch,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    g_delta, g_gamma, g_theta, g_vega, g_rho = await run_sync(
                        engine.price_batch_greeks,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    
                    for idx, (original_idx, req) in enumerate(items):
                        results[original_idx] = PriceResult(
                            price=float(prices[idx]),
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=float(g_delta[idx]),
                                gamma=float(g_gamma[idx]),
                                theta=float(g_theta[idx]),
                                vega=float(g_vega[idx]),
                                rho=float(g_rho[idx]),
                            )
                        )
                elif model_type == "monte_carlo":
                    # Monte Carlo batch pricing via Numba Parallel
                    prices = await run_sync(
                        engine.price_batch,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    g_delta, g_gamma, g_theta, g_vega, g_rho = await run_sync(
                        engine.price_batch_greeks,
                        spots, strikes, maturities, vols, rates, divs, types
                    )
                    
                    for idx, (original_idx, req) in enumerate(items):
                        results[original_idx] = PriceResult(
                            price=float(prices[idx]),
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=float(g_delta[idx]),
                                gamma=float(g_gamma[idx]),
                                theta=float(g_theta[idx]),
                                vega=float(g_vega[idx]),
                                rho=float(g_rho[idx]),
                            )
                        )
                else:
                    # Fallback to concurrent scalar pricing for non-vectorized engines
                    for original_idx, req in items:
                        params = req.to_bs_params()
                        res = await run_sync(engine.price_european, params, req.option_type)
                        results[original_idx] = PriceResult(
                            price=res.price,
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=res.greeks.delta,
                                gamma=res.greeks.gamma,
                                theta=res.greeks.theta,
                                vega=res.greeks.vega,
                                rho=res.greeks.rho,
                            ) if res.greeks else None
                        )

            except Exception as exc:
                logger.error("group_pricing_failed", model=model_type, error=str(exc))
                for original_idx, req in items:
                    results[original_idx] = PriceResult(
                        price=0.0, spot=req.spot, strike=req.strike,
                        time_to_expiry=req.time_to_expiry, rate=req.rate,
                        volatility=req.volatility, option_type=req.option_type,
                        model=model_type, computation_time_ms=0.0, greeks=None
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


    async def calculate_greeks_batch(self, options: list[Any]) -> BatchGreeksResult:
        """
        HIGH-PERFORMANCE: Batch Greeks calculation.
        """
        start_time = time.perf_counter()
        
        # Truly vectorized parameters
        spots = np.array([o.spot for o in options], dtype=np.float64)
        strikes = np.array([o.strike for o in options], dtype=np.float64)
        maturities = np.array([o.time_to_expiry for o in options], dtype=np.float64)
        vols = np.array([o.volatility for o in options], dtype=np.float64)
        rates = np.array([o.rate for o in options], dtype=np.float64)
        divs = np.array([o.dividend_yield for o in options], dtype=np.float64)
        types = np.array([o.option_type for o in options])

        from src.pricing.black_scholes import BlackScholesEngine
        from src.api.schemas.pricing import GreeksResult

        # Using BlackScholesEngine truly vectorized batch greeks (Rust/JIT)
        g_res = await run_sync(
            BlackScholesEngine.calculate_greeks,
            spots, strikes, maturities, vols, rates, divs, types
        )
        
        results = [
            GreeksResult(
                delta=float(g_res.delta[i]),
                gamma=float(g_res.gamma[i]),
                theta=float(g_res.theta[i]),
                vega=float(g_res.vega[i]),
                rho=float(g_res.rho[i]),
                option_price=0.0, # Price omitted for pure Greeks call
                spot=options[i].spot,
                strike=options[i].strike,
                time_to_expiry=options[i].time_to_expiry,
                volatility=options[i].volatility,
                option_type=options[i].option_type,
            )
            for i in range(len(options))
        ]

        return BatchGreeksResult(
            results=results,
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def calculate_iv_batch(self, options: list[Any]) -> list[float]:
        """
        Vectorized batch IV calculation.
        """
        if not options:
            return []

        market_prices = np.array([o.market_price for i, o in enumerate(options)], dtype=np.float64)
        spots = np.array([o.spot for i, o in enumerate(options)], dtype=np.float64)
        strikes = np.array([o.strike for i, o in enumerate(options)], dtype=np.float64)
        maturities = np.array([o.time_to_expiry for i, o in enumerate(options)], dtype=np.float64)
        rates = np.array([o.rate for i, o in enumerate(options)], dtype=np.float64)
        dividends = np.array([o.dividend_yield for i, o in enumerate(options)], dtype=np.float64)
        option_types = np.array([o.option_type for i, o in enumerate(options)])

        from src.pricing.implied_vol import vectorized_implied_volatility

        vols = await run_sync(
            vectorized_implied_volatility,
            market_prices,
            spots,
            strikes,
            maturities,
            rates,
            dividends,
            option_types,
        )
        return [float(v) for v in vols]
pricing_service = PricingService()
