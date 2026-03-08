import asyncio
import os
import time

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
from src.utils.ray_pool import RayActorPool
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

# Initialize Global Pool
_pool: RayActorPool | None = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = RayActorPool(MathActor, name="math_v2")
    return _pool


@app.task(base=BaseAsyncTask, bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> dict:
    """Non-blocking calibration delegation using BaseAsyncTask loop."""
    try:
        # EXECUTE: In a shared persistent loop via BaseAsyncTask
        return self.run_async(_recalibrate_symbols_batch_impl([symbol]))[0]
    except Exception as e:
        logger.error("calibration_task_failed", symbol=symbol, error=str(e))
        raise self.retry(exc=e) from e


@app.task(base=BaseAsyncTask, bind=True)
def recalibrate_symbols_batch(self, symbols: list[str]) -> list[dict]:
    """🚀 GOD-MODE: Non-blocking batch calibration delegation."""
    try:
        return self.run_async(_recalibrate_symbols_batch_impl(symbols))
    except Exception as e:
        logger.error("batch_calibration_task_failed", symbols=symbols, error=str(e))
        return []


async def _recalibrate_symbols_batch_impl(symbols: list[str]) -> list[dict]:
    """Async implementation of batch calibration."""
    try:
        # 1. Fetch market data snapshots in parallel
        router = MarketDataRouter()
        snapshots = await asyncio.gather(*[router.get_option_chain_snapshot(s) for s in symbols])

        valid_symbols = []
        valid_data = []
        for s, data in zip(symbols, snapshots):
            if data:
                valid_symbols.append(s)
                valid_data.append(data)

        if not valid_symbols:
            return []

        # 2. Delegate to Ray Actor Pool
        pool = get_pool()
        actor = await pool.get_actor()
        
        # 🚀 GOD-MODE: Non-blocking await of Ray future
        return await actor.run_calibration_batch.remote(valid_symbols, valid_data)

    except Exception as exc:
        logger.error("batch_calib_impl_error", symbols=symbols, error=str(exc))
        return []


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
    """Implementation of risk state synchronization (Multi-Dimensional)."""
    from src.shared.shm_mesh import RiskStateBuffer
    from src.utils.cache import get_redis

    redis = get_redis()
    if not redis:
        return

    # 1. Atomic Fetch from Redis using Pipeline (Delta, Gamma, Vega)
    async with redis.pipeline(transaction=False) as pipe:
        pipe.get("portfolio_net_delta")
        pipe.get("portfolio_net_gamma")
        pipe.get("portfolio_net_vega")
        pipe.get("portfolio_margin_usage")
        results = await pipe.execute()

    # Results mapping: [delta, gamma, vega, margin]
    current_metrics = [float(r) if r is not None else 0.0 for r in results]
    
    # 2. Update SHM (The engine's local truth)
    try:
        risk_buf = RiskStateBuffer(create=False)
        # Optimized view-based update
        # Assuming RiskStateBuffer.view is [d, g, v, max_d, max_g, max_v, margin, ts]
        # We update the metrics and ensure limits are synced from settings
        risk_buf.update(
            current_metrics[0], # delta
            current_metrics[1], # gamma
            current_metrics[2], # vega
            settings.MAX_NET_DELTA,
            settings.MAX_NET_GAMMA,
            settings.MAX_NET_VEGA,
            current_metrics[3]  # margin
        )
        logger.debug("risk_vector_synced", delta=current_metrics[0], gamma=current_metrics[1])
    except Exception as e:
        logger.error("shm_vector_update_failed", error=str(e))


_calibration_worker = _recalibrate_symbol_fallback
