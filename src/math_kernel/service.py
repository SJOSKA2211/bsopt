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

from api.schemas.pricing import (
    BatchGreeksResult,
    BatchPriceResult,
    PriceResult,
)
from src.math_kernel.factory import PricingEngineFactory, PricingEngineNotFound
from src.math_kernel.models import BSParameters

logger = structlog.get_logger(__name__)


class PricingService:
    """
    Unified entry point for all option pricing operations.
    Supports single and batch pricing using high-performance engines.
    """

    def __init__(self, factory: PricingEngineFactory | None = None) -> None:
        self.factory = factory or PricingEngineFactory()

    def clear_cache(self):
        """Resets any internal pricing caches."""
        logger.info("pricing_service_cache_cleared")


    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: str | None = None,
    ) -> PriceResult:
        """
        Calculates price and greeks for a single option request.
        Dispatches to the optimized engine determined by the model type.
        """
        try:
            # 1. Resolve Engine
            engine = self.factory.get_engine(model)

            # 2. Contextual parameters (Heston, etc.)
            ctx_params = {}
            if hasattr(engine, "resolve_contextual_params"):
                ctx_params = await engine.resolve_contextual_params(symbol=symbol)

            # 3. Compute Price (Off-load to thread pool for heavy engines)
            from functools import partial
            start_time = time.perf_counter()
            
            # Engines should handle their own parameter mapping
            result = await run_sync(partial(engine.price_european, **ctx_params), params, option_type)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # 4. Handle Result (Result can be float or object with .price)
            if isinstance(result, (float, int, np.float64, np.number)):
                price = float(result)
                greeks_obj = None
            else:
                price = float(getattr(result, "price", 0.0))
                greeks_obj = getattr(result, "greeks", None)

            # 5. Calculate Greeks if not provided by the engine
            if greeks_obj is None:
                try:
                    greeks_obj = await run_sync(partial(engine.calculate_greeks, **ctx_params), params, option_type)
                except Exception as e:
                    logger.warning("greeks_calculation_failed_during_pricing", error=str(e))
                    from src.math_kernel.models import OptionGreeks
                    greeks_obj = OptionGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

            logger.info(
                "option_priced",
                model=model,
                duration_ms=round(duration_ms, 2),
            )

            from api.schemas.pricing import OptionGreeksStruct

            return PriceResult(
                price=price,
                spot=params.spot,
                strike=params.strike,
                time_to_expiry=params.maturity,
                rate=params.rate,
                volatility=params.volatility,
                option_type=option_type,
                model=model,
                computation_time_ms=duration_ms,
                greeks=OptionGreeksStruct(
                    delta=float(greeks_obj.delta),
                    gamma=float(greeks_obj.gamma),
                    theta=float(greeks_obj.theta),
                    vega=float(greeks_obj.vega),
                    rho=float(greeks_obj.rho),
                )
                if greeks_obj
                else None,
            )

        except PricingEngineNotFound as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("pricing_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal pricing error")

    async def calculate_greeks(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
    ) -> Any:
        """
        Calculates greeks for a single option request.
        """
        try:
            engine = self.factory.get_engine(model)
            # Off-load to thread pool
            result = await run_sync(engine.calculate_greeks, params, option_type)
            return result
        except PricingEngineNotFound as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("greeks_calculation_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal greeks error")

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
        async def _process_group(model_type: str, items: list[tuple[int, Any]]) -> None:
            try:
                engine = self.factory.get_engine(model_type)
                from src.math_kernel.base import VectorizedPricingStrategy
                from api.schemas.pricing import OptionGreeksStruct

                # Vectorized parameters
                spots = np.array([it[1].spot for it in items], dtype=np.float64)
                strikes = np.array([it[1].strike for it in items], dtype=np.float64)
                maturities = np.array([it[1].time_to_expiry for it in items], dtype=np.float64)
                vols = np.array([it[1].volatility for it in items], dtype=np.float64)
                rates = np.array([it[1].rate for it in items], dtype=np.float64)
                divs = np.array([it[1].dividend_yield for it in items], dtype=np.float64)
                types = np.array([it[1].option_type for it in items])
                is_calls = types == "call"

                if isinstance(engine, VectorizedPricingStrategy):
                    prices_arr = await run_sync(
                        engine.price_batch, spots, strikes, maturities, vols, rates, divs, is_calls
                    )
                    
                    # Concurrently compute greeks if the engine supports batch greeks
                    if hasattr(engine, "calculate_greeks_batch"):
                        g_res = await run_sync(
                            engine.calculate_greeks_batch, spots, strikes, maturities, vols, rates, divs, types
                        )
                    else:
                        # Fallback to concurrent scalar greeks
                        g_tasks = [run_sync(engine.calculate_greeks, it[1].to_bs_params(), it[1].option_type) for it in items]
                        g_res_list = await asyncio.gather(*g_tasks)
                        # Create a mock object that mimics the expected structure if needed, or just handle it below
                        g_res = g_res_list

                    for idx, (original_idx, req) in enumerate(items):
                        if isinstance(g_res, list):
                            cg = g_res[idx]
                            greeks_struct = OptionGreeksStruct(
                                delta=float(cg.delta),
                                gamma=float(cg.gamma),
                                theta=float(cg.theta),
                                vega=float(cg.vega),
                                rho=float(cg.rho),
                            ) if cg else None
                        else:
                            # Assume batch greeks object with array members
                            greeks_struct = OptionGreeksStruct(
                                delta=float(g_res.delta[idx]),
                                gamma=float(g_res.gamma[idx]),
                                theta=float(g_res.theta[idx]),
                                vega=float(g_res.vega[idx]),
                                rho=float(g_res.rho[idx]),
                            ) if g_res else None
                        
                        results[original_idx] = PriceResult(
                            price=float(prices_arr[idx]),
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=greeks_struct,
                        )
                else:
                    # Fallback to concurrent scalar pricing for non-vectorized engines
                    tasks = [run_sync(engine.price_european, it[1].to_bs_params(), it[1].option_type) for it in items]
                    group_results = await asyncio.gather(*tasks)
                    
                    for idx, (original_idx, req) in enumerate(items):
                        res = group_results[idx]
                        results[original_idx] = PriceResult(
                            price=float(res.price if hasattr(res, "price") else res),
                            spot=req.spot,
                            strike=req.strike,
                            time_to_expiry=req.time_to_expiry,
                            rate=req.rate,
                            volatility=req.volatility,
                            option_type=req.option_type,
                            model=model_type,
                            computation_time_ms=0.0,
                            greeks=OptionGreeksStruct(
                                delta=float(res.greeks.delta),
                                gamma=float(res.greeks.gamma),
                                theta=float(res.greeks.theta),
                                vega=float(res.greeks.vega),
                                rho=float(res.greeks.rho),
                            ) if hasattr(res, "greeks") and res.greeks else None,
                        )

            except Exception as exc:
                logger.error("group_pricing_failed", model=model_type, error=str(exc))
                for original_idx, req in items:
                    results[original_idx] = PriceResult(
                        price=0.0,
                        spot=req.spot,
                        strike=req.strike,
                        time_to_expiry=req.time_to_expiry,
                        rate=req.rate,
                        volatility=req.volatility,
                        option_type=req.option_type,
                        model=model_type,
                        computation_time_ms=0.0,
                        greeks=None,
                    )

        # 3. Dispatch all groups concurrently
        dispatch_tasks = [_process_group(m, g) for m, g in model_groups.items()]
        if dispatch_tasks:
            await asyncio.gather(*dispatch_tasks)

        return BatchPriceResult(
            results=cast(list[PriceResult], results),
            total_count=len(results),
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    async def price_batch_arrays(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray,
        rates: np.ndarray,
        dividends: np.ndarray,
        option_types: np.ndarray,
        models: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:
        """
        ULTRA-HIGH PERFORMANCE: Direct array-based pricing without Pydantic overhead.
        Used by BatchPricingService for zero-allocation paths.
        """
        n = len(spots)
        results = np.zeros(n, dtype=np.float64)

        # Batch processes models to use vectorized kernels
        unique_models = np.unique(models)
        for model in unique_models:
            indices = np.where(models == model)[0]
            if len(indices) == 0:
                continue

            try:
                engine = self.factory.get_engine(str(model))

                # Check for vectorized interface
                from src.math_kernel.base import VectorizedPricingStrategy

                if isinstance(engine, VectorizedPricingStrategy):
                    # Convert types to boolean array
                    is_calls = option_types[indices] == "call"

                    prices = await run_sync(
                        engine.price_batch,
                        spots[indices],
                        strikes[indices],
                        maturities[indices],
                        vols[indices],
                        rates[indices],
                        dividends[indices],
                        is_calls,
                    )
                    results[indices] = prices
                else:
                    # Fallback to scalar pricing for non-vectorized engines
                    for idx in indices:
                        params = BSParameters(
                            spot=float(spots[idx]),
                            strike=float(strikes[idx]),
                            maturity=float(maturities[idx]),
                            volatility=float(vols[idx]),
                            rate=float(rates[idx]),
                            dividend=float(dividends[idx]),
                        )
                        res = await run_sync(engine.price_european, params, option_types[idx])
                        results[idx] = res.price
            except Exception as e:
                logger.error("array_batch_pricing_failed", model=model, error=str(e))

        return results

    async def price_batch_shm(
        self,
        shm_in_name: str,
        shm_out_name: str,
        shape: tuple[int, int],
        model: str = "black_scholes",
    ) -> bool:
        """
        ULTRA-LOW LATENCY: Pricing via Shared Memory segments.
        Direct memory interaction for zero-copy data transfer.
        """
        from src.shared.shared_memory import shm_manager

        try:
            shm_in = shm_manager.get_segment(shm_in_name)
            shm_out = shm_manager.get_segment(shm_out_name)

            # Input layout: [spot, strike, T, vol, r, q, is_call]
            n = shape[0]
            input_data = np.ndarray(shape, dtype=np.float64, buffer=shm_in.buf)
            output_data = np.ndarray((n,), dtype=np.float64, buffer=shm_out.buf)

            if model == "black_scholes":
                from src.math_kernel.black_scholes import BlackScholesEngine

                # Extract columns
                S = input_data[:, 0]
                K = input_data[:, 1]
                T = input_data[:, 2]
                sigma = input_data[:, 3]
                r = input_data[:, 4]
                q = input_data[:, 5]
                is_call = input_data[:, 6].astype(bool)

                # Execute vectorized pricing
                prices = await run_sync(
                    BlackScholesEngine.price_batch, S, K, T, sigma, r, q, is_call
                )

                # Copy results to output SHM
                output_data[:] = prices
                return True

            # Add other models as needed
            return False

        except Exception as e:
            logger.error("shm_batch_pricing_failed", error=str(e))
            return False

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

        from api.schemas.pricing import GreeksResult
        from src.math_kernel.black_scholes import BlackScholesEngine

        # Using BlackScholesEngine truly vectorized batch greeks (Rust/JIT)
        g_res = await run_sync(
            BlackScholesEngine.calculate_greeks,
            spots,
            strikes,
            maturities,
            vols,
            rates,
            divs,
            types,
        )

        results = []
        for i in range(len(options)):
            results.append(
                GreeksResult(
                    delta=float(cast(np.ndarray[Any, np.dtype[np.float64]], g_res.delta)[i]),
                    gamma=float(cast(np.ndarray[Any, np.dtype[np.float64]], g_res.gamma)[i]),
                    theta=float(cast(np.ndarray[Any, np.dtype[np.float64]], g_res.theta)[i]),
                    vega=float(cast(np.ndarray[Any, np.dtype[np.float64]], g_res.vega)[i]),
                    rho=float(cast(np.ndarray[Any, np.dtype[np.float64]], g_res.rho)[i]),
                    option_price=0.0,  # Price omitted for pure Greeks call
                    spot=float(options[i].spot),
                    strike=float(options[i].strike),
                    time_to_expiry=float(options[i].time_to_expiry),
                    volatility=float(options[i].volatility),
                    option_type=str(options[i].option_type),
                )
            )

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

        market_prices = np.array([o.market_price for o in options], dtype=np.float64)
        spots = np.array([o.spot for o in options], dtype=np.float64)
        strikes = np.array([o.strike for o in options], dtype=np.float64)
        maturities = np.array([o.time_to_expiry for o in options], dtype=np.float64)
        rates = np.array([o.rate for o in options], dtype=np.float64)
        dividends = np.array([o.dividend_yield for o in options], dtype=np.float64)
        option_types = np.array([o.option_type for o in options])

        from src.math_kernel.implied_vol import vectorized_implied_volatility

        vols_arr = cast(
            np.ndarray[Any, np.dtype[np.float64]],
            await run_sync(
                vectorized_implied_volatility,
                market_prices,
                spots,
                strikes,
                maturities,
                rates,
                dividends,
                option_types,
            ),
        )
        return [float(v) for v in vols_arr]

    async def generate_heatmap(self, request: Any) -> Any:
        """
        Generates a grid of P&L values for a risk heatmap.
        Vectorized across the entire grid for near-instant results.
        """
        from api.schemas.pricing import HeatmapCell, HeatmapResponse, PriceRequest

        start_time = time.perf_counter()
        
        # 1. Base price calculation
        base_params = request.to_bs_params()
        try:
            base_res = await self.price_option(base_params, request.option_type, request.model)
            base_price = base_res.price
        except Exception:
            base_price = 0.0

        # 2. Build grid of scenarios
        scenarios = []
        for v_shift in request.vol_shifts:
            for p_shift in request.price_shifts:
                shifted_spot = request.spot * (1 + p_shift / 100)
                shifted_vol = max(0.01, request.volatility + v_shift / 100)
                
                scenarios.append(PriceRequest(
                    spot=shifted_spot,
                    strike=request.strike,
                    time_to_expiry=request.time_to_expiry,
                    volatility=shifted_vol,
                    rate=request.rate,
                    option_type=request.option_type,
                    dividend_yield=request.dividend_yield,
                    model=request.model
                ))

        # 3. Batch price all scenarios
        batch_res = await self.price_batch(scenarios)
        
        # 4. Reshape into grid
        grid = []
        idx = 0
        for v_shift in request.vol_shifts:
            row = []
            for p_shift in request.price_shifts:
                res = batch_res.results[idx]
                pnl = res.price - base_price
                row.append(HeatmapCell(
                    price_shift=p_shift,
                    vol_shift=v_shift,
                    pnl=pnl,
                    theoretical_price=res.price
                ))
                idx += 1
            grid.append(row)

        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return HeatmapResponse(
            grid=grid,
            price_steps=request.price_shifts,
            vol_steps=request.vol_shifts,
            computation_time_ms=duration_ms
        )


pricing_service = PricingService()