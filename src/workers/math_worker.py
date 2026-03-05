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

# 🥒 Global Persistent Loop and Actors (One per prefork process)
_loop = None
_actors = []


def get_event_loop():
    global _loop
    if _loop is None:
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
    return _loop


def get_actors():
    global _actors
    if not _actors:
        if ray.is_initialized():
            # Use detected cores from Ray
            resources = ray.cluster_resources()
            num_workers = int(resources.get("CPU", 2))
            _actors = [MathActor.remote() for _ in range(num_workers)]
            logger.info("ray_actors_initialized", count=len(_actors))
    return _actors


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Non-blocking calibration delegation."""
    loop = get_event_loop()
    try:
        # EXECUTE: In a shared process-level loop to avoid asyncio.run overhead
        return loop.run_until_complete(_recalibrate_symbol_impl(self, symbol))
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e) from e


async def _recalibrate_symbol_impl(self, self_task, symbol: str) -> dict:
    """Async implementation of calibration task."""
    try:
        # 1. Fetch market data (Async/Awaited)
        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        # 2. Delegate to Ray (Async/Awaited)
        import random

        actors = get_actors()
        if actors:
            # OPTIMIZED: Modern Ray async actor call
            actor = random.choice(actors)
            result = await actor.run_calibration.remote(symbol, market_data)
            return result

        # 3. Local Fallback (if Ray is unreachable)
        logger.warning("ray_unreachable_falling_back_local", symbol=symbol)
        return await _recalibrate_symbol_async(self_task, symbol, market_data)

    except Exception as exc:
        logger.error("calibration_impl_error", symbol=symbol, error=str(exc))
        raise exc


async def _recalibrate_symbol_async(self, symbol: str, market_data: list | None = None) -> dict:
    """Fallback Local Calibration (Shared logic with async impl)."""
    start_time = time.time()
    try:
        if market_data is None:
            router = MarketDataRouter()
            market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        # Run CPU-bound task in default executor to avoid blocking the loop
        loop = asyncio.get_event_loop()
        calibrator = HestonCalibrator()

        params, metrics = await loop.run_in_executor(
            None, lambda: calibrator.calibrate(market_data, symbol=symbol)
        )

        duration = time.time() - start_time
        CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
        
        return {
            "symbol": symbol,
            "status": "success",
            "params": {
                "kappa": params.kappa,
                "theta": params.theta,
                "sigma": params.sigma,
                "rho": params.rho,
                "v0": params.v0,
            },
            "metrics": metrics,
        }

    except Exception as exc:
        logger.error("calibration_fallback_error", symbol=symbol, error=str(exc))
        raise exc


def health_check() -> bool:
    return True


_calibration_worker = _recalibrate_symbol_async
