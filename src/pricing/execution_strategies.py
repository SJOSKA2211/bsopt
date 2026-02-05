import asyncio
import concurrent.futures
from abc import ABC, abstractmethod

import numpy as np

try:
    import ray
except ImportError:
    class RayMock:
        def put(self, obj): return obj
        def get(self, obj): return obj
    ray = RayMock()
import structlog

from src.config import settings
from src.utils.shared_memory import shm_manager

logger = structlog.get_logger(__name__)

class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    @abstractmethod
    async def execute(self, inputs: dict[str, np.ndarray], executor: concurrent.futures.ProcessPoolExecutor | None = None) -> np.ndarray:
        pass

class SequentialStrategy(ExecutionStrategy):
    """Sequential execution (fallback)."""
    async def execute(self, inputs: dict[str, np.ndarray], executor: concurrent.futures.ProcessPoolExecutor | None = None) -> np.ndarray:
        # Import inside to avoid circular deps
        from src.pricing.black_scholes import BlackScholesEngine
        n = len(inputs['spots'])
        out_prices = np.empty(n, dtype=np.float64)
        
        # We can still use the vectorized call, just not parallelized via pool
        BlackScholesEngine.price_options(
            inputs['spots'], inputs['strikes'], inputs['maturities'],
            inputs['vols'], inputs['rates'], inputs['dividends'],
            inputs['is_call'], out=out_prices
        )
        return out_prices

class RayStrategy(ExecutionStrategy):
    """Ray distributed execution."""
    async def execute(self, inputs: dict[str, np.ndarray], executor: concurrent.futures.ProcessPoolExecutor | None = None) -> np.ndarray:
        from src.services.pricing_service import _ray_worker_pricing
        
        s_ref = ray.put(inputs['spots'])
        k_ref = ray.put(inputs['strikes'])
        t_ref = ray.put(inputs['maturities'])
        v_ref = ray.put(inputs['vols'])
        r_ref = ray.put(inputs['rates'])
        q_ref = ray.put(inputs['dividends'])
        c_ref = ray.put(inputs['is_call'])
        
        ray_future = _ray_worker_pricing.remote(s_ref, k_ref, t_ref, v_ref, r_ref, q_ref, c_ref)
        return await asyncio.wrap_future(ray_future) if asyncio.iscoroutine(ray_future) else ray.get(ray_future)

class SHMStrategy(ExecutionStrategy):
    """Shared Memory + ProcessPoolExecutor execution."""
    async def execute(self, inputs: dict[str, np.ndarray], executor: concurrent.futures.ProcessPoolExecutor | None = None) -> np.ndarray:
        from src.services.pricing_service import _worker_shared_memory_pricing
        
        n = len(inputs['spots'])
        input_data = np.stack([
            inputs['spots'], inputs['strikes'], inputs['maturities'],
            inputs['vols'], inputs['rates'], inputs['dividends'],
            inputs['is_call'].astype(np.float64)
        ], axis=1)
        
        shm_in_name = shm_manager.acquire()
        shm_out_name = shm_manager.acquire()
        
        if not (shm_in_name and shm_out_name):
            # Fallback to sequential if SHM exhausted
            logger.warning("shm_exhausted_fallback_sequential")
            return await SequentialStrategy().execute(inputs)

        try:
            from src.utils.shm_worker import SHMContextManager
            
            with SHMContextManager(shm_in_name) as shms:
                shm_in = shms[0]
                np.ndarray(input_data.shape, dtype=np.float64, buffer=shm_in.buf)[:] = input_data[:]
            
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                executor, 
                _worker_shared_memory_pricing, 
                shm_in_name, input_data.shape, shm_out_name
            )
            
            if success is True:
                with SHMContextManager(shm_out_name) as shms:
                    shm_out = shms[0]
                    return np.ndarray((n,), dtype=np.float64, buffer=shm_out.buf).copy()
            else:
                raise RuntimeError(f"SHM Worker failed: {success}")
        finally:
            shm_manager.release(shm_in_name)
            shm_manager.release(shm_out_name)

class WASMStrategy(ExecutionStrategy):
    """WebAssembly execution strategy."""
    async def execute(self, inputs: dict[str, np.ndarray], executor: concurrent.futures.ProcessPoolExecutor | None = None) -> np.ndarray:
        from src.pricing.wasm_engine import WASMPricingEngine
        engine = WASMPricingEngine()
        # WASM engine expects arrays and returns a list/array
        res = engine.batch_price_black_scholes(
            inputs['spots'], inputs['strikes'], inputs['maturities'],
            inputs['vols'], inputs['rates'], inputs['dividends'],
            inputs['is_call']
        )
        return np.array(res, dtype=np.float64)

class StrategyFactory:
    @staticmethod
    def get_strategy(count: int, ray_active: bool) -> ExecutionStrategy:
        from src.pricing.wasm_engine import WASM_AVAILABLE
        
        if ray_active and count > settings.PRICING_LARGE_BATCH_THRESHOLD:
            return RayStrategy()
        elif count > settings.PRICING_LARGE_BATCH_THRESHOLD:
            return SHMStrategy()
        elif WASM_AVAILABLE:
            return WASMStrategy()
        else:
            return SequentialStrategy()
