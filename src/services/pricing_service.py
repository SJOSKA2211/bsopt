import orjson
import os
import time
import re
import asyncio
import concurrent.futures
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Union, cast, Tuple
from anyio.to_thread import run_sync
import structlog
import numpy as np
from cachetools import TTLCache
import ray # SOTA: Distributed Computing

# Ray initialization moved to src/api/main.py for lifecycle management

@ray.remote
def _ray_worker_pricing(spots, strikes, maturities, vols, rates, dividends, is_call):
    """Ray remote worker for zero-copy distributed pricing."""
    from src.pricing.quant_utils import batch_bs_price_jit
    n = len(spots)
    outputs = np.empty(n, dtype=np.float64)
    batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=outputs)
    return outputs

@ray.remote
def _ray_worker_heston(spots, strikes, maturities, rates, params_list, is_calls):
    """🚀 SINGULARITY: Vectorized Ray worker for Heston FFT pricing."""
    from src.pricing.models.heston_fft import batch_heston_price_jit
    n = len(spots)
    outputs = np.empty(n, dtype=np.float64)
    
    # 🚀 SOTA: Extract parameters into arrays for vectorized JIT kernel
    v0s = np.array([p.get('v0', 0.04) for p in params_list])
    kappas = np.array([p.get('kappa', 2.0) for p in params_list])
    thetas = np.array([p.get('theta', 0.04) for p in params_list])
    sigmas = np.array([p.get('sigma', 0.3) for p in params_list])
    rhos = np.array([p.get('rho', -0.7) for p in params_list])
    is_calls_arr = np.array(is_calls, dtype=bool)
    
    # Execute the parallel JIT kernel on the Ray node
    batch_heston_price_jit(
        spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, 
        is_calls_arr, outputs
    )
    return outputs

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
from src.utils.shared_memory import shm_manager
from src.shared.off_heap_logger import omega_logger
from src.shared.shm_mesh import SharedMemoryRingBuffer
from src.config import settings

logger = structlog.get_logger(__name__)

# Singleton reader for the Zero-Copy Market Mesh
_shm_reader = SharedMemoryRingBuffer(create=False)
_last_shm_head = 0

# Security constants
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{1,10}$")
MAX_BATCH_COMPLEXITY = 5000000  # Total complexity budget for pricing in a single batch

def _worker_shared_memory_pricing(shm_name_inputs, input_shape, shm_name_output):
    """
    Worker function for ProcessPoolExecutor to perform zero-allocation pricing.
    Reads from and writes to shared memory blocks to avoid pickling overhead.
    """
    try:
        from src.utils.shm_worker import SHMContextManager
        
        with SHMContextManager(shm_name_inputs, shm_name_output) as shms:
            existing_shm_in = shms[0]
            existing_shm_out = shms[1]
            
            # Reconstruct numpy arrays from buffers
            # Layout: [spot, strike, T, vol, r, q, is_call]
            inputs = np.ndarray(input_shape, dtype=np.float64, buffer=existing_shm_in.buf)
            outputs = np.ndarray((input_shape[0],), dtype=np.float64, buffer=existing_shm_out.buf)
            
            # Extract columns
            spots = inputs[:, 0]
            strikes = inputs[:, 1]
            maturities = inputs[:, 2]
            vols = inputs[:, 3]
            rates = inputs[:, 4]
            dividends = inputs[:, 5]
            is_call = inputs[:, 6].astype(bool)
            
            # Import inside worker to avoid serializing heavy modules
            from src.pricing.quant_utils import batch_bs_price_jit
            
            # Execute JIT kernel
            batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=outputs)
            
        return True
    except Exception as e:
        return str(e)

def _worker_shared_memory_greeks(shm_name_inputs, input_shape, shm_names_outputs):
    """
    Worker function for ProcessPoolExecutor to perform zero-allocation Greeks calculation.
    """
    try:
        from src.utils.shm_worker import SHMContextManager
        
        with SHMContextManager(shm_name_inputs, shm_names_outputs) as shms:
            # shms[0] is input
            # shms[1..6] are outputs in order (delta, gamma, vega, theta, rho, price)
            
            existing_shm_in = shms[0]
            shm_delta = shms[1]
            shm_gamma = shms[2]
            shm_vega = shms[3]
            shm_theta = shms[4]
            shm_rho = shms[5]
            shm_price = shms[6]
            
            inputs = np.ndarray(input_shape, dtype=np.float64, buffer=existing_shm_in.buf)
            n = input_shape[0]
            
            out_delta = np.ndarray((n,), dtype=np.float64, buffer=shm_delta.buf)
            out_gamma = np.ndarray((n,), dtype=np.float64, buffer=shm_gamma.buf)
            out_vega = np.ndarray((n,), dtype=np.float64, buffer=shm_vega.buf)
            out_theta = np.ndarray((n,), dtype=np.float64, buffer=shm_theta.buf)
            out_rho = np.ndarray((n,), dtype=np.float64, buffer=shm_rho.buf)
            out_price = np.ndarray((n,), dtype=np.float64, buffer=shm_price.buf)
            
            # Extract columns
            spots = inputs[:, 0]
            strikes = inputs[:, 1]
            maturities = inputs[:, 2]
            vols = inputs[:, 3]
            rates = inputs[:, 4]
            dividends = inputs[:, 5]
            is_call = inputs[:, 6].astype(bool)
            
            from src.pricing.quant_utils import batch_bs_price_jit, batch_greeks_jit
            
            # Execute JIT kernels
            batch_greeks_jit(
                spots, strikes, maturities, vols, rates, dividends, is_call,
                out_delta=out_delta, out_gamma=out_gamma, out_vega=out_vega,
                out_theta=out_theta, out_rho=out_rho
            )
            batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=out_price)
            
        return True
    except Exception as e:
        return str(e)

def _worker_shared_memory_heston(shm_name_inputs, input_shape, shm_name_output):
    """
    Worker function for Heston model pricing using shared memory and Numba.
    Input Layout: [spot, strike, T, r, v0, kappa, theta, sigma, rho, is_call] (10 columns)
    """
    try:
        from src.utils.shm_worker import SHMContextManager
        
        with SHMContextManager(shm_name_inputs, shm_name_output) as shms:
            existing_shm_in, existing_shm_out = shms
        
            inputs = np.ndarray(input_shape, dtype=np.float64, buffer=existing_shm_in.buf)
            outputs = np.ndarray((input_shape[0],), dtype=np.float64, buffer=existing_shm_out.buf)
            
            # SOTA: Use the vectorized Numba integrator directly
            from src.pricing.models.heston_fft import _simpson_integral_jit
            
            # Extract columns for vectorization
            spots = inputs[:, 0]
            strikes = inputs[:, 1]
            maturities = inputs[:, 2]
            rates = inputs[:, 3]
            v0s = inputs[:, 4]
            kappas = inputs[:, 5]
            thetas = inputs[:, 6]
            sigmas = inputs[:, 7]
            rhos = inputs[:, 8]
            is_calls = inputs[:, 9] > 0.5
            
            # OPTIMIZATION: Moving the integration loop into a helper JIT function 
            # to avoid Python overhead during row-by-row processing.
            from src.pricing.models.heston_fft import batch_heston_price_jit
            batch_heston_price_jit(
                spots, strikes, maturities, rates, v0s, kappas, thetas, sigmas, rhos, is_calls, 
                out=outputs
            )
                
        return True
    except Exception as e:
        return str(e)

def _worker_shared_memory_mc(shm_name_inputs, input_shape, shm_name_output):
    """Worker for Monte Carlo batch pricing."""
    try:
        from src.utils.shm_worker import SHMContextManager
        
        with SHMContextManager(shm_name_inputs, shm_name_output) as shms:
            shm_in = shms[0]
            shm_out = shms[1]
            
            inputs = np.ndarray(input_shape, dtype=np.float64, buffer=shm_in.buf)
            outputs = np.ndarray((input_shape[0],), dtype=np.float64, buffer=shm_out.buf)
            from src.pricing.quant_utils import jit_mc_european_price
            
            for i in range(input_shape[0]):
                row = inputs[i] # [S, K, T, sigma, r, q, is_call, n_paths]
                price, _ = jit_mc_european_price(row[0], row[1], row[2], row[4], row[3], row[5], int(row[7]), row[6] > 0.5, True)
                outputs[i] = price
        return True
    except Exception as e: return str(e)

def _worker_single_heston(params_dict, spot, strike, maturity, rate, option_type):
    """Top-level worker for single Heston pricing to bypass GIL via ProcessPool."""
    from src.pricing.models.heston_fft import HestonModelFFT, HestonParams
    h_params = HestonParams(**params_dict)
    h_model = HestonModelFFT(h_params, r=rate, T=maturity)
    if option_type == "call":
        return h_model.price_call(spot, strike)
    else:
        return h_model.price_put(spot, strike)

class PricingService:
    """
    Unified Service for all Option Pricing operations.
    Consolidates logic from API routes and CLI.
    """
    _executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def __init__(self):
        # L1 Cache: Local memory LRU with 60s TTL for hot keys (max 1000 items)
        self._l1_cache = TTLCache(maxsize=1000, ttl=60)
        # SOTA: Ray handles the cluster/process scaling automatically
        self._ray_active = ray.is_initialized()
        
        # Initialize ProcessPoolExecutor for heavy GIL-bound tasks (Task 2)
        if PricingService._executor is None:
            # Optimal worker count for I/O + heavy JIT math
            worker_count = min(os.cpu_count() or 4, 8)
            PricingService._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=None # Default context
            )
            logger.info("initialized_process_pool_executor", workers=worker_count)

    def clear_cache(self):
        """Clear the L1 local cache."""
        self._l1_cache.clear()

    async def price_option(
        self,
        params: BSParameters,
        option_type: str,
        model: str = "black_scholes",
        symbol: Optional[str] = None,
        redis_client: Any = None
    ) -> PriceResponse:
        """
        Price a single option with off-heap logging and SHM mesh sync.
        """
        # Omega: Zero-latency hot-path logging
        omega_logger.log("PRICE_OPTION_START", symbol=symbol, model=model)
        
        start_time = time.perf_counter()
        
        # 0. Sync with Market Mesh (Zero-Copy)
        global _last_shm_head
        new_ticks, _last_shm_head = _shm_reader.read_latest(_last_shm_head)
        if new_ticks:
            # SOTA: Update local state if needed
            pass
        
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
                        cache_obj = orjson.loads(cached_data)
                        params_age = time.time() - cache_obj.get("timestamp", 0)
                        
                        if params_age <= 600:
                            h_params_dict = cache_obj["params"]
                            
                            # Offload to ProcessPoolExecutor to bypass GIL (Task 2)
                            loop = asyncio.get_event_loop()
                            price = await loop.run_in_executor(
                                self._executor, 
                                _worker_single_heston, 
                                h_params_dict, params.spot, params.strike, params.maturity, params.rate, option_type
                            )
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

    async def price_batch(self, options_data: List[PriceRequest], redis_client: Optional[Any] = None) -> BatchPriceResponse:
        """
        Ultra-high performance vectorized batch pricing.
        Leverages Numba JIT, ProcessPoolExecutor (for large batches), and SharedMemory.
        """
        if not options_data:
            return BatchPriceResponse(
                results=[], 
                total_count=0, 
                cached_count=0, 
                computation_time_ms=0.0
            )

        start_time = time.perf_counter()
        
        # 0. SOTA: Redis Pipelining for Heston parameters
        heston_param_map = {}
        heston_indices_all = [i for i, opt in enumerate(options_data) if getattr(opt, 'model', 'black_scholes') == 'heston']
        
        if heston_indices_all and redis_client:
            symbols = list(set(options_data[idx].symbol for idx in heston_indices_all if options_data[idx].symbol))
            if symbols:
                keys = [f"heston_params:{s}" for s in symbols]
                # Use MGET for atomic batch fetch
                try:
                    cached_values = await redis_client.mget(keys)
                    for symbol, val in zip(symbols, cached_values):
                        if val:
                            try:
                                cache_obj = orjson.loads(val)
                                # Only use if fresh (within 10 mins)
                                if time.time() - cache_obj.get("timestamp", 0) <= 600:
                                    heston_param_map[symbol] = cache_obj["params"]
                            except Exception:
                                continue
                except Exception as e:
                    logger.warning("heston_batch_cache_fetch_failed", error=str(e))

        # Split options into vectorized (BS), heavy (Heston), and sequential paths
        bs_indices = []
        heston_indices = []
        mc_indices = []
        fdm_indices = []
        other_indices = []
        for i, opt in enumerate(options_data):
            model = getattr(opt, 'model', 'black_scholes')
            if model == 'black_scholes':
                bs_indices.append(i)
            elif model == 'heston':
                heston_indices.append(i)
            elif model in ['monte_carlo', 'mc']:
                mc_indices.append(i)
            elif model in ['fdm', 'crank_nicolson']:
                fdm_indices.append(i)
            else:
                other_indices.append(i)
        
        final_results: List[Optional[PriceResponse]] = [None] * len(options_data)

        # WASM PRE-CHECK (Task 3)
        from src.pricing.wasm_engine import WASMPricingEngine, WASM_AVAILABLE
        wasm_engine = WASMPricingEngine() if WASM_AVAILABLE else None

        # 1. Vectorized Black-Scholes Path
        if bs_indices:
            n_bs = len(bs_indices)
            
            # Prepare inputs
            inputs = {
                'spots': np.array([options_data[i].spot for i in bs_indices], dtype=np.float64),
                'strikes': np.array([options_data[i].strike for i in bs_indices], dtype=np.float64),
                'maturities': np.array([options_data[i].time_to_expiry for i in bs_indices], dtype=np.float64),
                'vols': np.array([options_data[i].volatility for i in bs_indices], dtype=np.float64),
                'rates': np.array([options_data[i].rate for i in bs_indices], dtype=np.float64),
                'dividends': np.array([options_data[i].dividend_yield for i in bs_indices], dtype=np.float64),
                'is_call': np.array([options_data[i].option_type.lower() == "call" for i in bs_indices], dtype=bool)
            }
            
            from src.pricing.execution_strategies import StrategyFactory
            strategy = StrategyFactory.get_strategy(n_bs, self._ray_active)
            out_prices = await strategy.execute(inputs, self._executor)
            
            for i, idx in enumerate(bs_indices):
                final_results[idx] = self._format_price_response(float(out_prices[i]), options_data[idx], options_data[idx].option_type, "black_scholes", False, start_time)

        # 2. Parallel Heston Path (Task 2 & 4)
        if heston_indices:
            n_heston = len(heston_indices)
            
            # Try WASM Batch first (Task 3)
            wasm_success = False
            if wasm_engine and n_heston >= 1:
                try:
                    spots = np.array([options_data[i].spot for i in heston_indices], dtype=np.float64)
                    strikes = np.array([options_data[i].strike for i in heston_indices], dtype=np.float64)
                    maturities = np.array([options_data[i].time_to_expiry for i in heston_indices], dtype=np.float64)
                    rates = np.array([options_data[i].rate for i in heston_indices], dtype=np.float64)
                    
                    # For Heston, we need the parameters per option. WASM expects them.
                    # Currently WASMPricingEngine batch_price_heston takes a single HestonParams object.
                    # If symbols differ, we need to loop or handle it. 
                    # Assuming for now they might be the same or we use the first one's params as a simplified batch.
                    # Better: Check if all symbols are the same.
                    symbol_params = [heston_param_map.get(options_data[i].symbol) for i in heston_indices]
                    if all(p == symbol_params[0] for p in symbol_params) and symbol_params[0] is not None:
                        from src.pricing.models import HestonParams
                        hp = HestonParams(**symbol_params[0])
                        out_prices = wasm_engine.batch_price_heston(spots, strikes, maturities, rates, hp)
                        if len(out_prices) == n_heston:
                            for i, idx in enumerate(heston_indices):
                                final_results[idx] = self._format_price_response(float(out_prices[i]), options_data[idx], options_data[idx].option_type, "heston_wasm", False, start_time)
                            wasm_success = True
                except Exception as e:
                    logger.warning("wasm_batch_heston_failed", error=str(e))

            if not wasm_success:
                if self._ray_active and n_heston >= settings.PRICING_HESTON_PARALLEL_THRESHOLD:
                    # SOTA: Ray zero-copy distributed execution for heavy models
                    logger.info("using_ray_distributed_heston", count=n_heston)
                    
                    spots = np.array([options_data[i].spot for i in heston_indices], dtype=np.float64)
                strikes = np.array([options_data[i].strike for i in heston_indices], dtype=np.float64)
                maturities = np.array([options_data[i].time_to_expiry for i in heston_indices], dtype=np.float64)
                rates = np.array([options_data[i].rate for i in heston_indices], dtype=np.float64)
                is_calls = np.array([options_data[i].option_type.lower() == "call" for i in heston_indices], dtype=bool)
                
                params_list = [heston_param_map.get(options_data[i].symbol, {}) for i in heston_indices]
                
                # Remote execution
                ray_future = _ray_worker_heston.remote(spots, strikes, maturities, rates, params_list, is_calls)
                shared_out = ray.get(ray_future)
                
                for i, idx in enumerate(heston_indices):
                    final_results[idx] = self._format_price_response(float(shared_out[i]), options_data[idx], options_data[idx].option_type, "Heston-FFT-Ray", False, start_time)
            else:
                # Use shm_manager fallback
                input_data = np.zeros((n_heston, 10), dtype=np.float64)
                for i, idx in enumerate(heston_indices):
                    opt = options_data[idx]
                    p = heston_param_map.get(opt.symbol, {})
                    input_data[i] = [
                        opt.spot, opt.strike, opt.time_to_expiry, opt.rate,
                        p.get("v0", 0.04), p.get("kappa", 2.0), p.get("theta", 0.04),
                        p.get("sigma", 0.3), p.get("rho", -0.7),
                        1.0 if opt.option_type.lower() == "call" else 0.0
                    ]
                
                shm_in_name = shm_manager.acquire()
                shm_out_name = shm_manager.acquire()
                
                if shm_in_name and shm_out_name:
                    try:
                        shm_in = shm_manager.get_segment(shm_in_name)
                        shm_out = shm_manager.get_segment(shm_out_name)
                        shared_in = np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)
                        shared_in[:] = input_data[:]
                        loop = asyncio.get_event_loop()
                        success = await loop.run_in_executor(self._executor, _worker_shared_memory_heston, shm_in_name, input_data.shape, shm_out_name)
                        if success is True:
                            shared_out = np.ndarray((n_heston,), dtype=np.float64, buffer=shm_out.buf)
                            for i, idx in enumerate(heston_indices):
                                final_results[idx] = self._format_price_response(float(shared_out[i]), options_data[idx], options_data[idx].option_type, "Heston-FFT-Parallel", False, start_time)
                        else:
                            other_indices.extend(heston_indices)
                    finally:
                        shm_manager.release(shm_in_name)
                        shm_manager.release(shm_out_name)
                else:
                    # Fallback
                    shm_in = shared_memory.SharedMemory(create=True, size=input_data.nbytes)
                    shm_out = shared_memory.SharedMemory(create=True, size=n_heston * 8)
                    try:
                        shared_in = np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)
                        shared_in[:] = input_data[:]
                        loop = asyncio.get_event_loop()
                        success = await loop.run_in_executor(self._executor, _worker_shared_memory_heston, shm_in.name, input_data.shape, shm_out.name)
                        if success is True:
                            shared_out = np.ndarray((n_heston,), dtype=np.float64, buffer=shm_out.buf)
                            for i, idx in enumerate(heston_indices):
                                final_results[idx] = self._format_price_response(float(shared_out[i]), options_data[idx], options_data[idx].option_type, "Heston-FFT-Parallel", False, start_time)
                        else:
                            other_indices.extend(heston_indices) # Fallback to sequential
                    finally:
                        shm_in.close(); shm_in.unlink(); shm_out.close(); shm_out.unlink()

        # 3. Parallel Monte Carlo Path
        if mc_indices:
            n_mc = len(mc_indices)
            
            # Try WASM Batch first (Task 3)
            wasm_success = False
            if wasm_engine and n_mc >= 1:
                try:
                    spots = np.array([options_data[i].spot for i in mc_indices], dtype=np.float64)
                    strikes = np.array([options_data[i].strike for i in mc_indices], dtype=np.float64)
                    maturities = np.array([options_data[i].time_to_expiry for i in mc_indices], dtype=np.float64)
                    vols = np.array([options_data[i].volatility for i in mc_indices], dtype=np.float64)
                    rates = np.array([options_data[i].rate for i in mc_indices], dtype=np.float64)
                    dividends = np.array([options_data[i].dividend_yield for i in mc_indices], dtype=np.float64)
                    is_call = np.array([options_data[i].option_type.lower() == "call" for i in mc_indices], dtype=bool)
                    
                    out_prices = wasm_engine.batch_price_monte_carlo(spots, strikes, maturities, vols, rates, dividends, is_call)
                    if len(out_prices) == n_mc:
                        for i, idx in enumerate(mc_indices):
                            final_results[idx] = self._format_price_response(float(out_prices[i]), options_data[idx], options_data[idx].option_type, "monte_carlo_wasm", False, start_time)
                        wasm_success = True
                except Exception as e:
                    logger.warning("wasm_batch_mc_failed", error=str(e))

            if not wasm_success:
                # MC is very expensive, threshold is 1
                if n_mc >= 1:
                    input_data = np.zeros((n_mc, 8), dtype=np.float64)
                    for i, idx in enumerate(mc_indices):
                        opt = options_data[idx]
                        input_data[i] = [
                            opt.spot, opt.strike, opt.time_to_expiry, opt.volatility, opt.rate, 
                            opt.dividend_yield, 1.0 if opt.option_type.lower() == "call" else 0.0, 
                            float(settings.MONTE_CARLO_GPU_THRESHOLD) # Use configured paths threshold
                        ]
                    
                    shm_in_name = shm_manager.acquire()
                    shm_out_name = shm_manager.acquire()
                    
                    if shm_in_name and shm_out_name:
                        try:
                            shm_in = shm_manager.get_segment(shm_in_name)
                            shm_out = shm_manager.get_segment(shm_out_name)
                            shared_in = np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)
                            shared_in[:] = input_data[:]
                            loop = asyncio.get_event_loop()
                            success = await loop.run_in_executor(self._executor, _worker_shared_memory_mc, shm_in_name, input_data.shape, shm_out_name)
                            if success is True:
                                shared_out = np.ndarray((n_mc,), dtype=np.float64, buffer=shm_out.buf)
                                for i, idx in enumerate(mc_indices):
                                    final_results[idx] = self._format_price_response(float(shared_out[i]), options_data[idx], options_data[idx].option_type, "Monte-Carlo-Parallel", False, start_time)
                            else:
                                other_indices.extend(mc_indices)
                        finally:
                            shm_manager.release(shm_in_name)
                            shm_manager.release(shm_out_name)
                    else:
                        # Final fallback
                        other_indices.extend(mc_indices)
                else:
                    other_indices.extend(mc_indices)

        # 4. Parallel FDM Path
        if fdm_indices:
            n_fdm = len(fdm_indices)
            
            # Try WASM Batch first (Task 3)
            wasm_success = False
            if wasm_engine and n_fdm >= 1:
                try:
                    spots = np.array([options_data[i].spot for i in fdm_indices], dtype=np.float64)
                    strikes = np.array([options_data[i].strike for i in fdm_indices], dtype=np.float64)
                    maturities = np.array([options_data[i].time_to_expiry for i in fdm_indices], dtype=np.float64)
                    vols = np.array([options_data[i].volatility for i in fdm_indices], dtype=np.float64)
                    rates = np.array([options_data[i].rate for i in fdm_indices], dtype=np.float64)
                    dividends = np.array([options_data[i].dividend_yield for i in fdm_indices], dtype=np.float64)
                    is_call = np.array([options_data[i].option_type.lower() == "call" for i in fdm_indices], dtype=bool)
                    
                    out_prices = wasm_engine.batch_price_american_cn(spots, strikes, maturities, vols, rates, dividends, is_call)
                    if len(out_prices) == n_fdm:
                        for i, idx in enumerate(fdm_indices):
                            final_results[idx] = self._format_price_response(float(out_prices[i]), options_data[idx], options_data[idx].option_type, "fdm_cn_wasm", False, start_time)
                        wasm_success = True
                except Exception as e:
                    logger.warning("wasm_batch_fdm_failed", error=str(e))

            if not wasm_success:
                if n_fdm >= 1:
                    input_data = np.zeros((n_fdm, 9), dtype=np.float64)
                    for i, idx in enumerate(fdm_indices):
                        opt = options_data[idx]
                        input_data[i] = [opt.spot, opt.strike, opt.time_to_expiry, opt.volatility, opt.rate, opt.dividend_yield, 1.0 if opt.option_type.lower() == "call" else 0.0, 200.0, 200.0] # M=200, N=200
                    
                    shm_in_name = shm_manager.acquire()
                    shm_out_name = shm_manager.acquire()
                    
                    if shm_in_name and shm_out_name:
                        try:
                            shm_in = shm_manager.get_segment(shm_in_name)
                            shm_out = shm_manager.get_segment(shm_out_name)
                            shared_in = np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)
                            shared_in[:] = input_data[:]
                            loop = asyncio.get_event_loop()
                            success = await loop.run_in_executor(self._executor, _worker_shared_memory_fdm, shm_in_name, input_data.shape, shm_out_name)
                            if success is True:
                                shared_out = np.ndarray((n_fdm,), dtype=np.float64, buffer=shm_out.buf)
                                for i, idx in enumerate(fdm_indices):
                                    final_results[idx] = self._format_price_response(float(shared_out[i]), options_data[idx], options_data[idx].option_type, "FDM-CN-Parallel", False, start_time)
                            else:
                                other_indices.extend(fdm_indices)
                        finally:
                            shm_manager.release(shm_in_name)
                            shm_manager.release(shm_out_name)
                    else:
                        # Final fallback
                        other_indices.extend(fdm_indices)
                else:
                    other_indices.extend(fdm_indices)

        # 5. Concurrent Fallback Path
        if other_indices:
            async def _safe_price_option(idx: int):
                try:
                    opt = options_data[idx]
                    params = opt.to_bs_params()
                    return await self.price_option(params, opt.option_type, opt.model, opt.symbol)
                except Exception as e:
                    logger.warning("batch_option_failed", index=idx, error=str(e))
                    return None

            fallback_results = await asyncio.gather(*[_safe_price_option(idx) for idx in other_indices])
            for i, result in enumerate(fallback_results):
                if result:
                    final_results[other_indices[i]] = result


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
        Utilizes ProcessPoolExecutor and SharedMemory for large batches.
        """
        if not options_data:
            return BatchGreeksResponse(results=[], total_count=0, computation_time_ms=0.0)

        start_time = time.perf_counter()
        n = len(options_data)
        
        # Output buffers
        out_delta = np.empty(n, dtype=np.float64)
        out_gamma = np.empty(n, dtype=np.float64)
        out_vega = np.empty(n, dtype=np.float64)
        out_theta = np.empty(n, dtype=np.float64)
        out_rho = np.empty(n, dtype=np.float64)
        out_prices = np.empty(n, dtype=np.float64)

        if n > settings.PRICING_LARGE_BATCH_THRESHOLD:
            # MULTI-PROCESS PATH (Task 2 & 4)
            logger.info("using_process_pool_shared_memory_greeks", count=n)
            
            # Prepare input data grid
            input_data = np.zeros((n, 7), dtype=np.float64)
            for i, opt in enumerate(options_data):
                input_data[i] = [
                    opt.spot, opt.strike, opt.time_to_expiry, 
                    opt.volatility, opt.rate, opt.dividend_yield,
                    1.0 if opt.option_type.lower() == "call" else 0.0
                ]
            
            # Allocate shared memory using pool
            shm_in_name = shm_manager.acquire()
            shm_out_names = {
                'delta': shm_manager.acquire(),
                'gamma': shm_manager.acquire(),
                'vega': shm_manager.acquire(),
                'theta': shm_manager.acquire(),
                'rho': shm_manager.acquire(),
                'price': shm_manager.acquire()
            }
            
            # Filter out None if pool is exhausted
            if shm_in_name is None or any(v is None for v in shm_out_names.values()):
                # Release what we got and fallback
                if shm_in_name: shm_manager.release(shm_in_name)
                for v in shm_out_names.values():
                    if v: shm_manager.release(v)
                # Fallback to creating fresh segments if pool empty (less ideal but safe)
                shm_in = shared_memory.SharedMemory(create=True, size=input_data.nbytes)
                shm_names_out = {k: f"shm_{k}_{id(options_data)}" for k in ['delta', 'gamma', 'vega', 'theta', 'rho', 'price']}
                outputs_shm = {k: shared_memory.SharedMemory(create=True, name=name, size=n * 8) 
                              for k, name in shm_names_out.items()}
                shm_in_name = shm_in.name
                shm_names_out_final = shm_names_out
                is_pooled = False
            else:
                shm_in = shm_manager.get_segment(shm_in_name)
                outputs_shm = {k: shm_manager.get_segment(name) for k, name in shm_out_names.items()}
                shm_names_out_final = shm_out_names
                is_pooled = True
            
            try:
                shared_in = np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)
                shared_in[:] = input_data[:]
                
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self._executor, 
                    _worker_shared_memory_greeks, 
                    shm_in_name, input_data.shape, shm_names_out_final
                )
                
                if success is True:
                    out_delta[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['delta'].buf)[:]
                    out_gamma[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['gamma'].buf)[:]
                    out_vega[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['vega'].buf)[:]
                    out_theta[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['theta'].buf)[:]
                    out_rho[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['rho'].buf)[:]
                    out_prices[:] = np.ndarray((n,), dtype=np.float64, buffer=outputs_shm['price'].buf)[:]
                else:
                    raise RuntimeError(f"Parallel greeks error: {success}")
            finally:
                if is_pooled:
                    shm_manager.release(shm_in_name)
                    for v in shm_out_names.values():
                        shm_manager.release(v)
                else:
                    shm_in.close()
                    shm_in.unlink()
                    for shm in outputs_shm.values():
                        shm.close()
                        shm.unlink()
        else:
            # THREADPOOL PATH
            from src.pricing.black_scholes import BlackScholesEngine
            spots = np.array([o.spot for o in options_data], dtype=np.float64)
            strikes = np.array([o.strike for o in options_data], dtype=np.float64)
            maturities = np.array([o.time_to_expiry for o in options_data], dtype=np.float64)
            vols = np.array([o.volatility for o in options_data], dtype=np.float64)
            rates = np.array([o.rate for o in options_data], dtype=np.float64)
            dividends = np.array([o.dividend_yield for o in options_data], dtype=np.float64)
            types = np.array([o.option_type for o in options_data])

            await run_sync(
                BlackScholesEngine.calculate_greeks_batch,
                spots, strikes, maturities, vols, rates, dividends, types, None,
                out_delta, out_gamma, out_vega, out_theta, out_rho
            )
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
        redis_client: Optional[Any] = None
    ) -> np.ndarray:
        """
        Highest performance batch pricing path using raw numpy arrays.
        Bypasses PriceRequest object creation for zero-allocation.
        """
        n = len(spots)
        out_prices = np.empty(n, dtype=np.float64)
        
        # SOTA: Group by model for vectorized execution
        # For simplicity, if all are Black-Scholes (99% of batch cases)
        unique_models = np.unique(models)
        
        if len(unique_models) == 1 and unique_models[0] in ['black_scholes', 'bs']:
            # FAST PATH: Direct JIT or Parallel execution
            is_call = (option_types == 'call') | (option_types == 'CALL')
            
            if self._ray_active and n > settings.PRICING_LARGE_BATCH_THRESHOLD:
                ray_future = _ray_worker_pricing.remote(
                    ray.put(spots), ray.put(strikes), ray.put(maturities), 
                    ray.put(vols), ray.put(rates), ray.put(dividends), ray.put(is_call)
                )
                out_prices[:] = ray.get(ray_future)[:]
            elif n > settings.PRICING_LARGE_BATCH_THRESHOLD:
                # SOTA: Zero-allocation via SharedMemory for ProcessPool (Task 4)
                input_data = np.stack([spots, strikes, maturities, vols, rates, dividends, is_call.astype(np.float64)], axis=1)
                shm_in_name = shm_manager.acquire()
                shm_out_name = shm_manager.acquire()
                if shm_in_name and shm_out_name:
                    try:
                        shm_in = shm_manager.get_segment(shm_in_name)
                        shm_out = shm_manager.get_segment(shm_out_name)
                        np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)[:] = input_data[:]
                        loop = asyncio.get_event_loop()
                        success = await loop.run_in_executor(self._executor, _worker_shared_memory_pricing, shm_in_name, input_data.shape, shm_out_name)
                        if success is True:
                            out_prices[:] = np.ndarray((n,), dtype=np.float64, buffer=shm_out.buf)[:]
                        else:
                            from src.pricing.quant_utils import batch_bs_price_jit
                            batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=out_prices)
                    finally:
                        shm_manager.release(shm_in_name)
                        shm_manager.release(shm_out_name)
                else:
                    from src.pricing.quant_utils import batch_bs_price_jit
                    batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=out_prices)
            else:
                from src.pricing.quant_utils import batch_bs_price_jit
                batch_bs_price_jit(spots, strikes, maturities, vols, rates, dividends, is_call, out=out_prices)
        else:
            # MIXED PATH: Reuse existing price_batch logic by wrapping (temporary)
            # Better: Refactor price_batch to use this internal array logic
            from src.api.schemas.pricing import PriceRequest
            options_data = []
            for i in range(n):
                options_data.append(PriceRequest(
                    spot=spots[i], strike=strikes[i], time_to_expiry=maturities[i],
                    volatility=vols[i], rate=rates[i], dividend_yield=dividends[i],
                    option_type=option_types[i], model=models[i], symbol=symbols[i]
                ))
            batch_res = await self.price_batch(options_data, redis_client)
            for i, r in enumerate(batch_res.results):
                out_prices[i] = r.price
                
        return out_prices

    async def price_batch_shm(
        self,
        shm_name_in: str,
        shm_name_out: str,
        shape: Tuple[int, int],
        model: str = 'black_scholes'
    ) -> bool:
        """
        SOTA: Ultimate zero-allocation path.
        Expects data already written to shm_name_in.
        """
        if model in ['black_scholes', 'bs']:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, 
                _worker_shared_memory_pricing, 
                shm_name_in, shape, shm_name_out
            )
        elif model == 'heston':
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, 
                _worker_shared_memory_heston, 
                shm_name_in, shape, shm_name_out
            )
        return False

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
