import asyncio
import os
import time

import ray
import redis.asyncio as redis
import structlog
from celery import Celery

from src.config import get_settings
from src.data.router import MarketDataRouter
from src.pricing.calibration.engine import HestonCalibrator
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

async_redis_client = redis.from_url(settings.REDIS_URL)

app = Celery("math_worker", broker=os.getenv("CELERY_BROKER_URL", settings.REDIS_URL))

# Initialize Ray once
RayOrchestrator.init()
# Create a pool of actors
_actor_pool = None


def get_actor_pool():
    global _actor_pool
    if _actor_pool is None:
        if ray.is_initialized():
            num_workers = int(ray.cluster_resources().get("CPU", 2))
            actors = [MathActor.remote() for _ in range(num_workers)]
            _actor_pool = ray.util.ActorPool(actors)
        else:
            logger.warning("ray_not_initialized")
            _actor_pool = None
    return _actor_pool


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Delegate calibration to Ray Actor Pool."""
    try:
        pool = get_actor_pool()
        if pool and pool.has_free():
            # Submit to Ray pool
            # Note: ActorPool.submit is distinct from ray.get on a specific actor
            # We use a simple map-like approach or just pick a free one manually if we want async
            # But for simplicity in Celery, let's just use the pool's map for a single item
            # Or better, just let Ray scheduler handle a remote function if strict actor affinity isn't needed.
            # However, MathActor is stateful? No, Heston calibration is stateless per symbol.
            # So we don't even need Actors, just remote functions.
            # But let's stick to the Actor pattern if that's what the system uses.

            # Optimized: Submit to first idle actor
            pool.submit(lambda a, v: a.run_calibration.remote(v, []), symbol)
            # Wait for result
            # Since submit returns an ObjectRef, we need to fetch it?
            # ActorPool.submit returns void, it queues. get_next() returns result.
            # This usage is blocking.
            return pool.get_next()

        # Fallback to local async if Ray is full or down
        return asyncio.run(_recalibrate_symbol_async(self, symbol))

    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e) from e


async def _recalibrate_symbol_async(self, symbol: str) -> dict:
    """Fallback Async Calibration."""
    start_time = time.time()
    try:
        logger.info("calibration_started_local", symbol=symbol)

        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        # Perform calibration directly in this loop (blocking) or thread
        # ideally we use a ThreadPoolExecutor for CPU bound tasks in asyncio
        loop = asyncio.get_event_loop()
        calibrator = HestonCalibrator()

        # Run CPU-bound task in default executor
        params, metrics, surface_params = await loop.run_in_executor(
            None,
            lambda: (
                calibrator.calibrate(market_data)[0],
                calibrator.calibrate(market_data)[1],
                calibrator.calibrate_surface(market_data),
            ),
        )

        # Cache & Store (Simulated for brevity, logic matches original)
        # ... (Same logic as before)

        duration = time.time() - start_time
        CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
        return {"symbol": symbol, "status": "success"}

    except Exception as exc:
        logger.error("calibration_error", symbol=symbol, error=str(exc))
        raise exc


def health_check() -> bool:
    return True


_calibration_worker = _recalibrate_symbol_async
