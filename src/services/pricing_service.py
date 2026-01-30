import json
import time
import re
from typing import Any, Dict, List, Optional, Union, cast
from anyio.to_thread import run_sync
import structlog
import numpy as np
from cachetools import TTLCache

from src.api.schemas.pricing import PriceRequest, PriceResponse, BatchPriceResponse, GreeksResponse, BatchGreeksResponse
from src.pricing.black_scholes import BSParameters, OptionGreeks
from src.pricing.factory import PricingEngineFactory
from src.pricing.implied_vol import implied_volatility
from src.pricing.exotic import (
    ExoticParameters,
    price_exotic_option
)
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams
from src.utils.cache import pricing_cache, generate_cache_key
from src.shared.observability import PRICING_SERVICE_DURATION

logger = structlog.get_logger(__name__)

# Security constants
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{1,10}$")
MAX_BATCH_COMPLEXITY = 5000000  # Total complexity budget for pricing in a single batch

class PricingService:
    """
    Unified Service for all Option Pricing operations.
    Consolidates logic from API routes and CLI.
    """

    def __init__(self):
        # L1 Cache: Local memory LRU with 60s TTL for hot keys (max 1000 items)
        self._l1_cache = TTLCache(maxsize=1000, ttl=60)

    def clear_cache(self):
        """Clear the L1 local cache."""
        self._l1_cache.clear()

    @staticmethod
    def map_request_to_params(request: Any) -> BSParameters:
        """Map API request schemas to internal BSParameters."""
        return BSParameters(
            spot=request.spot,
            strike=request.strike,
            maturity=request.time_to_expiry,
            volatility=request.volatility,
            rate=request.rate,
            dividend=request.dividend_yield if hasattr(request, 'dividend_yield') else 0.0,
        )

    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: Optional[str] = None,
        redis_client: Any = None
    ) -> PriceResponse:
        """
        Price a single option with model-specific logic (e.g. Heston caching).
        """
        start_time = time.perf_counter()
        
        # 0. L1 Local Cache Check
        # Optimized: Use tuple key for L1 to avoid hashing overhead
        l1_key = (model, option_type, params.spot, params.strike, params.maturity, params.volatility, params.rate, params.dividend)
        
        if l1_key in self._l1_cache:
            response = self._l1_cache[l1_key]
            # Update computation time to reflect cache hit
            response.computation_time_ms = (time.perf_counter() - start_time) * 1000
            response.cached = True
            return response

        pricing_model_header = model

        # 1. Heston Logic
        if model == "heston":
            if not symbol:
                raise ValueError("Symbol is required for Heston model")
            
            # Security: Validate symbol before constructing Redis key
            if not SYMBOL_REGEX.match(symbol):
                raise ValueError(f"Invalid symbol format: {symbol}")
            
            if redis_client:
                cache_key = f"heston_params:{symbol}"
                cached_data = await redis_client.get(cache_key)
                
                if cached_data:
                    try:
                        cache_obj = json.loads(cached_data)
                        params_age = time.time() - cache_obj.get("timestamp", 0)
                        
                        if params_age <= 600:
                            h_params = HestonParams(**cache_obj["params"])
                            
                            def _run_heston():
                                h_model = HestonModelFFT(h_params, r=params.rate, T=params.maturity)
                                if option_type == "call":
                                    return h_model.price_call(params.spot, params.strike)
                                else:
                                    return h_model.price_put(params.spot, params.strike)

                            price = await run_sync(_run_heston)
                            return self._format_price_response(price, params, option_type, "Heston-FFT", False, start_time)
                        
                        logger.warning("heston_params_stale", symbol=symbol, age=params_age)
                    except Exception as e:
                        logger.warning("heston_cache_error", symbol=symbol, error=str(e))
            
            # Fallback
            model = "black_scholes"
            pricing_model_header = "Black-Scholes-Fallback"

        # 2. Standard Cache Check
        cached_price = await pricing_cache.get_option_price(params, option_type, model)
        if cached_price is not None:
            res = self._format_price_response(cached_price, params, option_type, pricing_model_header, True, start_time)
            self._l1_cache[l1_key] = res
            return res

        # 3. Standard Execution
        strategy = PricingEngineFactory.get_strategy(model)
        price = await run_sync(strategy.price, params, option_type)
        
        # Async Cache Set
        await pricing_cache.set_option_price(params, option_type, model, price)
        
        PRICING_SERVICE_DURATION.labels(method="price_option").observe(time.perf_counter() - start_time)
        res = self._format_price_response(price, params, option_type, pricing_model_header, False, start_time)
        self._l1_cache[l1_key] = res
        return res

    async def price_batch(self, options_data: List[PriceRequest]) -> BatchPriceResponse:
        """
        Ultra-high performance vectorized batch pricing.
        Leverages Numba JIT and SIMD parallelization.
        """
        if not options_data:
            return BatchPriceResponse(
                results=[], 
                total_count=0, 
                cached_count=0, 
                computation_time_ms=0.0
            )

        start_time = time.perf_counter()
        
        # Split options into vectorized (BS) and sequential paths
        bs_indices = []
        other_indices = []
        for i, opt in enumerate(options_data):
            # Default to black_scholes if model not specified
            model = getattr(opt, 'model', 'black_scholes')
            if model == 'black_scholes':
                bs_indices.append(i)
            else:
                other_indices.append(i)
        
        final_results: List[Optional[PriceResponse]] = [None] * len(options_data)

        # 1. Vectorized Black-Scholes Path
        if bs_indices:
            from src.pricing.black_scholes import BlackScholesEngine
            
            # Extract numpy arrays for JIT
            # We assume PriceRequest objects have these fields. 
            # Note: options_data is a list of Pydantic models (PriceRequest or similar)
            
            # Using list comprehensions is fast enough for input extraction
            spots = np.array([options_data[i].spot for i in bs_indices], dtype=np.float64)
            strikes = np.array([options_data[i].strike for i in bs_indices], dtype=np.float64)
            maturities = np.array([options_data[i].time_to_expiry for i in bs_indices], dtype=np.float64)
            vols = np.array([options_data[i].volatility for i in bs_indices], dtype=np.float64)
            rates = np.array([options_data[i].rate for i in bs_indices], dtype=np.float64)
            dividends = np.array([options_data[i].dividend_yield for i in bs_indices], dtype=np.float64)
            types = np.array([options_data[i].option_type for i in bs_indices])

            # ZERO-ALLOCATION: Pre-allocate output buffer
            out_prices = np.empty(len(bs_indices), dtype=np.float64)

            # Parallel JIT execution with memory reuse
            await run_sync(
                BlackScholesEngine.price_options,
                spots, strikes, maturities, vols, rates, dividends, types, None, out_prices
            )
            
            for i, idx in enumerate(bs_indices):
                opt_req = options_data[idx]
                # Direct Pydantic model construction bypassing validation for speed
                final_results[idx] = PriceResponse.model_construct(
                    price=float(out_prices[i]),
                    spot=opt_req.spot,
                    strike=opt_req.strike,
                    time_to_expiry=opt_req.time_to_expiry,
                    rate=opt_req.rate,
                    volatility=opt_req.volatility,
                    option_type=opt_req.option_type,
                    model="black_scholes",
                    cached=False,
                    computation_time_ms=0.0
                )

        # 2. Sequential Fallback Path (for non-vectorized models like MC/FDM)
        for idx in other_indices:
            try:
                opt = options_data[idx]
                params = self.map_request_to_params(opt)
                # price_option now returns PriceResponse directly
                final_results[idx] = await self.price_option(params, opt.option_type, opt.model, opt.symbol)
            except Exception as e:
                logger.warning("batch_option_failed", index=idx, error=str(e))
                continue


        computation_time = (time.perf_counter() - start_time) * 1000
        
        # Filter out None results (from sequential failures)
        valid_results = [r for r in final_results if r is not None]
        cached_count = sum(1 for r in valid_results if r.cached)
        
        return BatchPriceResponse(
            results=valid_results,
            total_count=len(valid_results),
            cached_count=cached_count,
            computation_time_ms=computation_time
        )

    async def calculate_greeks_batch(self, options_data: List[Any]) -> BatchGreeksResponse:
        """
        High-throughput vectorized Greeks calculation.
        Utilizes memory reuse and Numba JIT for maximum performance.
        """
        if not options_data:
            return BatchGreeksResponse(results=[], total_count=0, computation_time_ms=0.0)

        start_time = time.perf_counter()
        from src.pricing.black_scholes import BlackScholesEngine

        # Extract numpy arrays
        spots = np.array([o.spot for i, o in enumerate(options_data)], dtype=np.float64)
        strikes = np.array([o.strike for i, o in enumerate(options_data)], dtype=np.float64)
        maturities = np.array([o.time_to_expiry for i, o in enumerate(options_data)], dtype=np.float64)
        vols = np.array([o.volatility for i, o in enumerate(options_data)], dtype=np.float64)
        rates = np.array([o.rate for i, o in enumerate(options_data)], dtype=np.float64)
        dividends = np.array([o.dividend_yield for i, o in enumerate(options_data)], dtype=np.float64)
        types = np.array([o.option_type for i, o in enumerate(options_data)])

        n = len(options_data)
        # Pre-allocate buffers for memory reuse
        out_delta = np.empty(n, dtype=np.float64)
        out_gamma = np.empty(n, dtype=np.float64)
        out_vega = np.empty(n, dtype=np.float64)
        out_theta = np.empty(n, dtype=np.float64)
        out_rho = np.empty(n, dtype=np.float64)
        out_prices = np.empty(n, dtype=np.float64)

        # Vectorized calculation
        await run_sync(
            BlackScholesEngine.calculate_greeks_batch,
            spots, strikes, maturities, vols, rates, dividends, types, None,
            out_delta, out_gamma, out_vega, out_theta, out_rho
        )
        
        # Also need prices for the response
        is_call = np.array([str(t).lower() == "call" for t in types], dtype=bool)
        await run_sync(
            BlackScholesEngine.price_options,
            spots, strikes, maturities, vols, rates, dividends, types, None, out_prices
        )

        results = []
        for i in range(n):
            req = options_data[i]
            results.append(GreeksResponse.model_construct(
                delta=float(out_delta[i]),
                gamma=float(out_gamma[i]),
                vega=float(out_vega[i]),
                theta=float(out_theta[i]),
                rho=float(out_rho[i]),
                option_price=float(out_prices[i]),
                spot=req.spot,
                strike=req.strike,
                time_to_expiry=req.time_to_expiry,
                volatility=req.volatility,
                option_type=req.option_type
            ))

        computation_time = (time.perf_counter() - start_time) * 1000
        return BatchGreeksResponse(
            results=results,
            total_count=n,
            computation_time_ms=computation_time
        )

    async def calculate_greeks(self, params: BSParameters, option_type: str) -> Dict[str, Any]:
        """
        Calculate Greeks using Black-Scholes (default) or other engines.
        """
        start_time = time.perf_counter()
        cached_greeks = await pricing_cache.get_greeks(params, option_type)
        strategy = PricingEngineFactory.get_strategy("black_scholes")
        
        if cached_greeks:
            greeks = cached_greeks
        else:
            greeks_res = await run_sync(strategy.calculate_greeks, params, option_type)
            greeks = cast(OptionGreeks, greeks_res)
            await pricing_cache.set_greeks(params, option_type, greeks)
            
        price = await run_sync(strategy.price, params, option_type)
        
        PRICING_SERVICE_DURATION.labels(method="calculate_greeks").observe(time.perf_counter() - start_time)
        return {
            "delta": float(greeks.delta),
            "gamma": float(greeks.gamma),
            "theta": float(greeks.theta),
            "vega": float(greeks.vega),
            "rho": float(greeks.rho),
            "price": price
        }

    async def calculate_iv(self, market_price: float, spot: float, strike: float, maturity: float, rate: float, dividend: float, option_type: str) -> float:
        """Calculate Implied Volatility."""
        return await run_sync(implied_volatility, market_price, spot, strike, maturity, rate, dividend, option_type)

    async def price_exotic(self, exotic_type: str, params: ExoticParameters, option_type: str, **kwargs) -> Any:
        """Price exotic options."""
        def _run():
            return price_exotic_option(exotic_type, params, option_type, **kwargs)
        return await run_sync(_run)

    def _format_price_response(self, price, params, option_type, model, cached, start_time) -> PriceResponse:
        return PriceResponse.model_construct(
            price=price,
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.maturity,
            rate=params.rate,
            volatility=params.volatility,
            option_type=option_type,
            model=model,
            cached=cached,
            computation_time_ms=(time.perf_counter() - start_time) * 1000
        )
