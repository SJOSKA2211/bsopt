import os
import asyncio
import time
from celery import Celery
from celery.schedules import crontab
from celery.signals import task_failure, task_success
import structlog
import redis.asyncio as redis
import json
from typing import Dict, List
from src.pricing.calibration.engine import HestonCalibrator, MarketOption
from src.pricing.models.heston_fft import HestonParams
from src.data.router import MarketDataRouter
from src.shared.observability import push_metrics, setup_logging, tune_gc, HESTON_FELLER_MARGIN, CALIBRATION_DURATION, MODEL_RMSE, HESTON_R_SQUARED, HESTON_PARAMS_FRESHNESS
from src.config import get_settings
from src.database import get_async_db_context
from src.database.models import CalibrationResult

setup_logging()
tune_gc()
logger = structlog.get_logger()
settings = get_settings()

app = Celery('math_worker', broker=settings.RABBITMQ_URL)
app.conf.result_backend = settings.REDIS_URL
app.conf.task_serializer = 'json'
app.conf.result_serializer = 'json'
app.conf.accept_content = ['json']
app.conf.timezone = 'UTC'

# Persistent Async Redis Client
async_redis_client = redis.from_url(settings.REDIS_URL)
redis_client = async_redis_client

# ============================================================================
# PERIODIC TASKS SCHEDULE
# ============================================================================
app.conf.beat_schedule = {
    'calibrate-spy-every-5-minutes': {
        'task': 'src.workers.math_worker.recalibrate_symbol',
        'schedule': 300.0,
        'args': ('SPY',)
    },
    'calibrate-qqq-every-5-minutes': {
        'task': 'src.workers.math_worker.recalibrate_symbol',
        'schedule': 300.0,
        'args': ('QQQ',)
    },
    'health-check-every-minute': {
        'task': 'src.workers.math_worker.health_check',
        'schedule': 60.0
    },
}

# ============================================================================
# CALIBRATION TASKS
# ============================================================================
@app.task
def health_check() -> Dict:
    """Verify system health. Uses sync-to-async bridge for Celery simplicity if needed, but we define as async."""
    return asyncio.run(_health_check_async())

async def _health_check_async() -> Dict:
    symbols = ['SPY', 'QQQ']
    health_status = {}
    for symbol in symbols:
        cache_key = f"heston_params:{symbol}"
        val = await async_redis_client.get(cache_key)
        if not val:
            health_status[symbol] = 'missing'
        else:
            data = json.loads(val)
            age = time.time() - data.get('timestamp', 0)
            health_status[symbol] = 'healthy' if age <= 600 else 'stale'
    
    healthy_count = len([v for v in health_status.values() if v == 'healthy'])
    HESTON_PARAMS_FRESHNESS.labels(symbol='aggregate').set(healthy_count)
    return health_status

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> Dict:
    """Wrapper for async calibration."""
    return asyncio.run(_recalibrate_symbol_async(self, symbol))

async def _recalibrate_symbol_async(self, symbol: str) -> Dict:
    """
    Persistent async calibration logic.
    """
    start_time = time.time()
    try:
        logger.info("calibration_started", symbol=symbol)
        
        router = MarketDataRouter()
        market_data = await router.get_option_chain_snapshot(symbol)

        if not market_data:
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        market_options = [MarketOption(**opt) for opt in market_data]
        calibrator = HestonCalibrator()
        params, quality_metrics = calibrator.calibrate(market_options)
        surface_params = calibrator.calibrate_surface(market_options)

        # Store in Redis
        cache_value = {
            'params': params.__dict__,
            'surface': {str(k): list(v) for k, v in surface_params.items()},
            'metrics': quality_metrics,
            'timestamp': time.time()
        }
        await async_redis_client.setex(f"heston_params:{symbol}", 600, json.dumps(cache_value))
        
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
