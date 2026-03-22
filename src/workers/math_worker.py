import asyncio
import concurrent.futures
import os
import time
from typing import Any

import orjson
import ray
import redis.asyncio as redis
import structlog
from celery import Celery

from src.config import get_settings
from src.data.router import MarketDataRouter
from src.database import get_async_db_context
from src.database.models import CalibrationResult
from src.pricing.calibration.engine import HestonCalibrator
from src.pricing.models.heston_fft import HestonParams
from src.shared.observability import (
    CALIBRATION_DURATION,
    setup_logging,
    tune_gc,
)
from src.utils.distributed import RayOrchestrator
from src.workers.ray_workers import MathActor

# Optimized event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

setup_logging()
tune_gc()
logger = structlog.get_logger()
settings = get_settings()

# Initialize missing references
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
async_redis_client = redis.from_url(settings.REDIS_URL)

app = Celery("math_worker", broker=os.getenv("CELERY_BROKER_URL", settings.REDIS_URL))

# Initialize Ray Swarm
# RayOrchestrator.init() -> Moved to lazy load
# math_swarm = [MathActor.remote() for _ in range(os.cpu_count() or 2)]
_math_swarm = None

def get_math_swarm():
    """Lazy initialize the Ray swarm."""
    global _math_swarm
    if _math_swarm is None:
        RayOrchestrator.init()
        # Check if we are in a test/mock environment where Ray might be mocked
        if ray.is_initialized():
             # Respect the capped CPU count from RayOrchestrator
             num_workers = int(ray.cluster_resources().get("CPU", 2))
             _math_swarm = [MathActor.remote() for _ in range(num_workers)]
        else:
             # Fallback for when Ray is mocked but not "initialized" in a way that allows remote()
             # Or if initialization failed silently.
             logger.warning("ray_not_initialized_in_swarm_getter")
             _math_swarm = []
    return _math_swarm

def _calibration_worker(market_data: Any) -> tuple[HestonParams, dict, dict]:
    """
    Worker function to be executed in a ProcessPoolExecutor.
    Performs heavy math calibration using HestonCalibrator.
    """
    calibrator = HestonCalibrator()
    params, metrics = calibrator.calibrate(market_data)
    surface = calibrator.calibrate_surface(market_data)
    return params, metrics, surface

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Delegate calibration to the optimal Ray Actor."""
    try:
        # Simple round-robin or Ray's internal scheduler can be used here
        swarm = get_math_swarm()
        if not swarm:
             raise RuntimeError("Ray swarm not available")
        actor = swarm[0] 
        result = ray.get(actor.run_calibration.remote(symbol, []))
        return result
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        # If Ray fails, fallback to the async local implementation
        return asyncio.run(_recalibrate_symbol_async(self, symbol))

async def _recalibrate_symbol_async(self, symbol: str) -> dict:
    """
    Persistent async calibration logic utilizing ProcessPoolExecutor for heavy math.
    """
    start_time = time.time()
    try:
        logger.info("calibration_started", symbol=symbol)
        
        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        # Offload heavy calibration to ProcessPoolExecutor (Task 2)
        loop = asyncio.get_event_loop()
        params, quality_metrics, surface_params = await loop.run_in_executor(
            executor, _calibration_worker, market_data
        )

        # Store in Redis
        cache_value = {
            'params': params.__dict__,
            'surface': {str(k): list(v) for k, v in surface_params.items()},
            'metrics': quality_metrics,
            'timestamp': time.time()
        }
        await async_redis_client.setex(f"heston_params:{symbol}", 600, orjson.dumps(cache_value))
        
        # Persist to PostgreSQL (Async)
        async with get_async_db_context() as db:
            db_res = CalibrationResult(
                symbol=symbol,
                v0=params.v0, kappa=params.kappa, theta=params.theta, 
                sigma=params.sigma, rho=params.rho,
                rmse=quality_metrics['rmse'],
                r_squared=quality_metrics['r_squared'],
                num_options=quality_metrics['num_options'],
                svi_params=cache_value['surface']
            )
            db.add(db_res)
            await db.commit()
        
        duration = time.time() - start_time
        CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
        logger.info("calibration_complete", symbol=symbol, rmse=quality_metrics['rmse'])
        return {'symbol': symbol, 'status': 'success'}
        
    except Exception as exc:
        logger.error("calibration_error", symbol=symbol, error=str(exc))
        if hasattr(self, "retry"):
            raise self.retry(exc=exc, countdown=60)
        raise exc

def health_check() -> bool:
    """Check if the math worker and its dependencies are healthy."""
    try:
        # Check Ray
        if not ray.is_initialized():
            return False
        return True
    except Exception:
        return False
