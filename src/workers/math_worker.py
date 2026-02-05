import asyncio
import os
import time

import orjson
import structlog
from celery import Celery

from src.config import get_settings
from src.data.router import MarketDataRouter
from src.database import get_async_db_context
from src.database.models import CalibrationResult
from src.shared.observability import (
    CALIBRATION_DURATION,
    setup_logging,
    tune_gc,
)

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

from src.utils.distributed import RayOrchestrator
from src.workers.ray_workers import MathActor

app = Celery("math_worker", broker=os.getenv("CELERY_BROKER_URL", settings.REDIS_URL))

# 🚀 SINGULARITY: Initialize Ray Hive Mind
RayOrchestrator.init()
math_swarm = [MathActor.remote() for _ in range(os.cpu_count() or 2)]

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Delegate calibration to the optimal Ray Actor."""
    try:
        # Simple round-robin or Ray's internal scheduler can be used here
        actor = math_swarm[0] 
        result = ray.get(actor.run_calibration.remote(symbol, []))
        return result
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e)
    """Wrapper for async calibration."""
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
        
        duration = time.time() - start_time
        CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
        logger.info("calibration_complete", symbol=symbol, rmse=quality_metrics['rmse'])
        return {'symbol': symbol, 'status': 'success'}
        
    except Exception as exc:
        logger.error("calibration_error", symbol=symbol, error=str(exc))
        raise self.retry(exc=exc, countdown=60)

def health_check() -> bool:
    """Check if the math worker and its dependencies are healthy."""
    try:
        # Check Ray
        import ray
        if not ray.is_initialized():
            return False
        return True
    except Exception:
        return False
