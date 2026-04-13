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
from src.database import get_async_db_context
from src.database.models import CalibrationResult
from src.ingestion.router import MarketDataRouter
from src.math_kernel.calibration.engine import HestonCalibrator
from src.math_kernel.models.heston_fft import HestonParams
from src.shared.observability import (
    CALIBRATION_DURATION,
    setup_logging,
    tune_gc,
)
from src.shared.utils.distributed import RayOrchestrator
from src.workers.ray_workers import MathActor

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

setup_logging()
tune_gc()
logger = structlog.get_logger()
settings = get_settings()

executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
async_redis_client = redis.from_url(settings.REDIS_URL)

app = Celery("math_worker", broker=os.getenv("CELERY_BROKER_URL", settings.REDIS_URL))

_math_swarm = None

def get_math_swarm():
    global _math_swarm
    if _math_swarm is None:
        RayOrchestrator.init()
        num_workers = int(ray.cluster_resources().get("CPU", 2))
        _math_swarm = [MathActor.remote() for _ in range(num_workers)]
    return _math_swarm

def _calibration_worker(market_data: Any) -> tuple[HestonParams, dict, dict]:
    calibrator = HestonCalibrator()
    params, metrics = calibrator.calibrate(market_data)
    surface = calibrator.calibrate_surface(market_data)
    return params, metrics, surface

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Orchestrate calibration via Ray Swarm with async fallback."""
    try:
        swarm = get_math_swarm()
        if not swarm:
            raise RuntimeError("Ray swarm unavailable")
        return ray.get(swarm[0].run_calibration.remote(symbol, []))
    except Exception as e:
        logger.error("ray_calibration_failed", symbol=symbol, error=str(e))
        return asyncio.run(_recalibrate_symbol_async(self, symbol))

async def _recalibrate_symbol_async(self, symbol: str) -> dict:
    """Core calibration logic using ProcessPool for parallelism."""
    start_time = time.time()
    try:
        logger.info("calibration_started", symbol=symbol)
        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        loop = asyncio.get_event_loop()
        params, quality_metrics, surface_params = await loop.run_in_executor(
            executor, _calibration_worker, market_data
        )

        # Persistence & Cache
        cache_value = {
            "params": params.__dict__,
            "surface": {str(k): list(v) for k, v in surface_params.items()},
            "metrics": quality_metrics,
            "timestamp": time.time(),
        }
        await async_redis_client.setex(f"heston_params:{symbol}", 600, orjson.dumps(cache_value))

        async with get_async_db_context() as db:
            db.add(CalibrationResult(
                symbol=symbol,
                v0=params.v0, kappa=params.kappa, theta=params.theta,
                sigma=params.sigma, rho=params.rho,
                rmse=quality_metrics["rmse"], r_squared=quality_metrics["r_squared"],
                num_options=quality_metrics["num_options"], svi_params=cache_value["surface"]
            ))
            await db.commit()

        CALIBRATION_DURATION.labels(symbol=symbol).observe(time.time() - start_time)
        return {"symbol": symbol, "status": "success"}

    except Exception as exc:
        logger.error("calibration_critical_error", symbol=symbol, error=str(exc))
        if self and hasattr(self, "retry"):
            raise self.retry(exc=exc, countdown=60)
        raise exc

def health_check() -> bool:
    try:
        return ray.is_initialized()
    except Exception:
        return False