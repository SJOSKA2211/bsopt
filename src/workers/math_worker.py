import os
import asyncio
import time
from celery import Celery
from celery.schedules import crontab
from celery.signals import task_failure, task_success
import structlog
import redis
import json
from typing import Dict, List
from src.pricing.calibration.engine import HestonCalibrator, MarketOption
from src.pricing.models.heston_fft import HestonParams
from src.data.router import MarketDataRouter
from src.shared.observability import push_metrics, setup_logging, HESTON_FELLER_MARGIN, CALIBRATION_DURATION, MODEL_RMSE, HESTON_R_SQUARED, HESTON_PARAMS_FRESHNESS
from src.config import get_settings
from src.database import get_db_context
from src.database.models import CalibrationResult

setup_logging()
logger = structlog.get_logger()
settings = get_settings()

setup_logging()
logger = structlog.get_logger()
settings = get_settings()

app = Celery('math_worker', broker=settings.RABBITMQ_URL)
app.conf.result_backend = settings.REDIS_URL
app.conf.task_serializer = 'json'
app.conf.result_serializer = 'json'
app.conf.accept_content = ['json']
app.conf.timezone = 'UTC'

redis_client = redis.from_url(settings.REDIS_URL)

# ============================================================================
# PERIODIC TASKS SCHEDULE
# ============================================================================
app.conf.beat_schedule = {
    'calibrate-spy-every-5-minutes': {
        'task': 'src.workers.math_worker.recalibrate_symbol',
        'schedule': 300.0,  # 5 minutes
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
# SIGNAL HANDLERS
# ============================================================================
@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Log task failures to monitoring system."""
    logger.error("task_failed", task=sender.name, task_id=task_id, error=str(exception))

# ============================================================================
# CALIBRATION TASKS
# ============================================================================
@app.task
def health_check() -> Dict:
    """Verify system health and parameter freshness."""
    symbols = ['SPY', 'QQQ']
    health_status = {}
    for symbol in symbols:
        cache_key = f"heston_params:{symbol}"
        val = redis_client.get(cache_key)
        if not val:
            health_status[symbol] = 'missing'
            logger.warning("params_missing", symbol=symbol)
        else:
            data = json.loads(val)
            age = time.time() - data.get('timestamp', 0)
            if age > 600:
                health_status[symbol] = 'stale'
                logger.warning("params_stale", symbol=symbol, age=age)
            else:
                health_status[symbol] = 'healthy'
    
    healthy_count = len([v for v in health_status.values() if v == 'healthy'])
    HESTON_PARAMS_FRESHNESS.labels(symbol='aggregate').set(healthy_count)
    return health_status

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def recalibrate_symbol(self, symbol: str) -> Dict:
    """
    Periodic calibration task.
    1. Fetch live option chain
    2. Run Heston calibration
    3. Store parameters in Redis with TTL
    """
    start_time = time.time()
    try:
        logger.info("calibration_started", symbol=symbol, task_id=self.request.id)
        
        # Step 1: Fetch market data
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        router = MarketDataRouter()
        market_data = loop.run_until_complete(router.get_option_chain_snapshot(symbol))

        if not market_data or not isinstance(market_data, list):
            # Fallback for empty/mock responses
            logger.warning("insufficient_data", symbol=symbol)
            return {"symbol": symbol, "status": "failed", "reason": "no_data"}

        market_options = [MarketOption(**opt) for opt in market_data]
        # Step 2: Run calibration
        calibrator = HestonCalibrator()
        params, quality_metrics = calibrator.calibrate(market_options)
        
        # Fit SVI surface
        surface_params = calibrator.calibrate_surface(market_options)

        # Step 3: Store in Redis
        cache_key = f"heston_params:{symbol}"
        cache_value = {
            'params': {
                'v0': params.v0,
                'kappa': params.kappa,
                'theta': params.theta,
                'sigma': params.sigma,
                'rho': params.rho
            },
            'surface': {str(k): list(v) for k, v in surface_params.items()},
            'metrics': quality_metrics,
            'timestamp': time.time(),
            'version': '1.0'
        }
        redis_client.setex(cache_key, 600, json.dumps(cache_value))
        
        # Step 4: Persist to PostgreSQL
        try:
            with get_db_context() as db:
                db_res = CalibrationResult(
                    symbol=symbol,
                    v0=params.v0,
                    kappa=params.kappa,
                    theta=params.theta,
                    sigma=params.sigma,
                    rho=params.rho,
                    rmse=quality_metrics['rmse'],
                    r_squared=quality_metrics['r_squared'],
                    num_options=quality_metrics['num_options'],
                    svi_params=cache_value['surface']
                )
                db.add(db_res)
        except Exception as e:
            logger.error("db_persistence_failed", symbol=symbol, error=str(e))
            # Continue even if DB fails, as Redis is the primary source for live pricing
        
        # 📊 METRICS
        duration = time.time() - start_time
        CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
        MODEL_RMSE.labels(model_type='heston', dataset='live').set(quality_metrics['rmse'])
        HESTON_R_SQUARED.labels(symbol=symbol).set(quality_metrics['r_squared'])
        HESTON_PARAMS_FRESHNESS.labels(symbol=symbol).set(time.time())
        feller_margin = 2 * params.kappa * params.theta - params.sigma**2
        HESTON_FELLER_MARGIN.labels(symbol=symbol).set(feller_margin)

        logger.info("calibration_complete", symbol=symbol, rmse=quality_metrics['rmse'])
        push_metrics("math_worker")
        
        return {
            'symbol': symbol,
            'status': 'success',
            'params': cache_value['params'],
            'metrics': quality_metrics
        }
    except Exception as exc:
        logger.error("calibration_error", symbol=symbol, error=str(exc))
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
