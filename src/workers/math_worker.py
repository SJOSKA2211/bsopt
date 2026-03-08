import asyncio
import os
import threading
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

from src.utils.celery import BaseAsyncTask

setup_logging()
tune_gc()
logger = structlog.get_logger()
settings = get_settings()

_async_redis_client: redis.Redis | None = None


def get_async_redis_client() -> redis.Redis:
    """Get or initialize the global async Redis client."""
    global _async_redis_client
    if _async_redis_client is None:
        _async_redis_client = redis.from_url(settings.REDIS_URL)
    return _async_redis_client


app = Celery("math_worker", broker=os.getenv("CELERY_BROKER_URL", settings.REDIS_URL))

# Initialize Ray once
RayOrchestrator.init()

class RayActorPool:
    """God-Mode Ray Actor Pool: Handles round-robin load balancing and health checks."""
    def __init__(self, actor_class, count: int | None = None):
        self._actor_class = actor_class
        self._count = count or int(ray.cluster_resources().get("CPU", 2))
        self._actors = [actor_class.remote() for _ in range(self._count)]
        self._index = 0
        self._lock = threading.Lock()
        logger.info("ray_actor_pool_initialized", count=self._count, actor=actor_class.__name__)

    def get_actor(self):
        with self._lock:
            actor = self._actors[self._index % self._count]
            self._index += 1
            return actor

# Initialize Global Pool
_pool: RayActorPool | None = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = RayActorPool(MathActor)
    return _pool


@app.task(base=BaseAsyncTask, bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Non-blocking calibration delegation using BaseAsyncTask loop."""
    try:
        # EXECUTE: In a shared persistent loop via BaseAsyncTask
        return self.run_async(_recalibrate_symbol_impl(symbol))
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e) from e


async def _recalibrate_symbol_impl(symbol: str) -> dict:
    """Async implementation of calibration task."""
    try:
        # 1. Fetch market data (Async/Awaited)
        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        # 2. Delegate to Ray (Async/Awaited)
        pool = get_pool()
        actor = pool.get_actor()
        
        # Ray async actor call
        result = await actor.run_calibration.remote(symbol, market_data)
        return result

    except Exception as exc:
        logger.error("calibration_impl_error", symbol=symbol, error=str(exc))
        # Fallback Local Calibration if Ray fails
        return await _recalibrate_symbol_fallback(symbol, None)


async def _recalibrate_symbol_fallback(symbol: str, market_data: list | None = None) -> dict:
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


@app.task(base=BaseAsyncTask, bind=True)
def reconcile_risk_state(self):
    """Periodically syncs Redis 'truth' to SHM RiskStateBuffer."""
    try:
        return self.run_async(_reconcile_risk_state_impl())
    except Exception as e:
        logger.error("risk_reconciliation_failed", error=str(e))


async def _reconcile_risk_state_impl():
    """Implementation of risk state synchronization."""
    from src.shared.shm_mesh import RiskStateBuffer
    from src.utils.cache import get_redis

    redis = get_redis()
    if not redis:
        return

    # 1. Fetch from Redis (The global truth)
    current_delta = await redis.get("portfolio_net_delta")
    if current_delta is None:
        return

    # 2. Update SHM (The engine's local truth)
    try:
        risk_buf = RiskStateBuffer(create=False)
        risk_buf.update(float(current_delta), settings.MAX_NET_DELTA)
        logger.debug("risk_shm_synced_from_redis", delta=current_delta)
    except Exception as e:
        logger.error("shm_update_failed", error=str(e))


_calibration_worker = _recalibrate_symbol_fallback
