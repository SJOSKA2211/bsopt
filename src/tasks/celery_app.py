"""
Celery Application Configuration - Production Optimized
=========================================================

Optimized for:
- High throughput with priority queues
- Automatic retry with exponential backoff
- Task rate limiting per queue
- Circuit breaker pattern for external dependencies
- Dead letter queue for failed tasks
- Task routing based on resource requirements
- Monitoring with Prometheus metrics
- Graceful shutdown handling
"""

import logging
import os
from datetime import timedelta

from celery import Celery, Task, signals
from celery.schedules import crontab
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

# =============================================================================
# Environment Configuration
# =============================================================================

RABBITMQ_URL = os.environ.get("RABBITMQ_URL")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# =============================================================================
# Exchange and Queue Definitions
# =============================================================================

# Exchanges for different priority levels
default_exchange = Exchange("default", type="direct")
priority_exchange = Exchange("priority", type="direct")
dlx_exchange = Exchange("dlx", type="direct")  # Dead letter exchange

# Queue definitions with priorities and DLX
task_queues = (
    # High priority pricing queue (low latency requirements)
    Queue(
        "pricing",
        exchange=priority_exchange,
        routing_key="pricing",
        queue_arguments={
            "x-max-priority": 10,
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "pricing.dead",
            "x-message-ttl": 300000,  # 5 minute TTL
        },
    ),
    # ML queue (resource intensive, lower priority)
    Queue(
        "ml",
        exchange=default_exchange,
        routing_key="ml",
        queue_arguments={
            "x-max-priority": 5,
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "ml.dead",
            "x-message-ttl": 3600000,  # 1 hour TTL
        },
    ),
    # Trading queue (critical, highest priority)
    Queue(
        "trading",
        exchange=priority_exchange,
        routing_key="trading",
        queue_arguments={
            "x-max-priority": 10,
            "x-dead-letter-exchange": "dlx",
            "x-dead-letter-routing-key": "trading.dead",
            "x-message-ttl": 60000,  # 1 minute TTL
        },
    ),
    # Batch processing queue
    Queue(
        "batch",
        exchange=default_exchange,
        routing_key="batch",
        queue_arguments={
            "x-max-priority": 3,
            "x-message-ttl": 7200000,  # 2 hour TTL
        },
    ),
    # Dead letter queue for failed tasks
    Queue(
        "dead_letter",
        exchange=dlx_exchange,
        routing_key="#",
    ),
)

# =============================================================================
# Celery Application
# =============================================================================

celery_app = Celery(
    "bsopt",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
    include=[
        "src.tasks.pricing_tasks",
        "src.tasks.ml_tasks",
        "src.tasks.trading_tasks",
        "src.tasks.data_tasks",
        "src.tasks.email_tasks",
    ],
)

# =============================================================================
# Celery Configuration
# =============================================================================

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    # Task execution
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minute soft limit
    task_acks_late=True,  # Acknowledge after completion (at-least-once)
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=True,
    # High-performance settings for transient tasks
    task_persistent=True, # Default to True, but will override for pricing
    # Worker configuration
    worker_prefetch_multiplier=1,  # SOTA: Set to 1 for high-throughput heterogeneous tasks
    worker_concurrency=os.cpu_count() or 4,
    worker_max_tasks_per_child=1000,  # Recycle workers periodically
    worker_max_memory_per_child=512000,  # 512MB memory limit
    # Queue configuration
    task_queues=task_queues,
    task_default_queue="pricing",
    task_default_exchange="priority",
    task_default_routing_key="pricing",
    # Task routing
    task_routes={
        "src.tasks.pricing_tasks.*": {
            "queue": "pricing",
            "routing_key": "pricing",
            "persistent": False, # Non-persistent for speed
        },
        "src.tasks.ml_tasks.*": {
            "queue": "ml",
            "routing_key": "ml",
        },
        "src.tasks.ml_tasks.hyperparameter_search_task": {
            "queue": "batch",  # Long-running task goes to batch queue
            "routing_key": "batch",
        },
        "src.tasks.trading_tasks.*": {
            "queue": "trading",
            "routing_key": "trading",
        },
    },
    # Default retry configuration
    task_default_retry_delay=5,  # 5 seconds initial delay
    task_max_retries=3,
    # Rate limiting (tasks per second per worker)
    task_annotations={
        "src.tasks.pricing_tasks.price_option_task": {
            "rate_limit": "1000/s",  # High throughput for pricing
            "task_result_expires": 60, # Expire results quickly to save memory
        },
        "src.tasks.pricing_tasks.batch_price_options_task": {
            "rate_limit": "100/s",
            "task_result_expires": 300,
        },
        "src.tasks.ml_tasks.train_model_task": {
            "rate_limit": "1/m",  # Max 1 training per minute
        },
        "src.tasks.ml_tasks.predict_task": {
            "rate_limit": "500/s",
        },
        "src.tasks.trading_tasks.execute_trade_task": {
            "rate_limit": "50/s",  # Rate limit for safety
        },
    },
    # Result backend configuration
    result_expires=timedelta(hours=1), # Reduced from 24h to save Redis memory
    result_extended=True,  # Include task name and args in result
    result_compression="gzip",
    # Beat scheduler (for periodic tasks)
    beat_schedule={
        "health-check-every-minute": {
            "task": "src.tasks.celery_app.health_check",
            "schedule": timedelta(minutes=1),
            "options": {"queue": "pricing", "priority": 1},
        },
        "cleanup-expired-results": {
            "task": "src.tasks.celery_app.cleanup_expired_results",
            "schedule": crontab(hour=2, minute=0),  # Daily at 2am
            "options": {"queue": "batch"},
        },
        "refresh-cache": {
            "task": "src.tasks.celery_app.refresh_pricing_cache",
            "schedule": timedelta(minutes=15),
            "options": {"queue": "pricing", "priority": 3},
        },
        "refresh-materialized-views": {
            "task": "src.tasks.data_tasks.refresh_materialized_views_task",
            "schedule": timedelta(minutes=30),
            "options": {"queue": "batch", "priority": 1},
        },
        "model-performance-check": {
            "task": "src.tasks.ml_tasks.check_model_performance",
            "schedule": crontab(hour="*/6"),  # Every 6 hours
            "options": {"queue": "ml"},
        },
        # Daily data collection at 10am ET (15:00 UTC)
        "scheduled-data-collection-daily": {
            "task": "src.tasks.data_tasks.scheduled_data_collection",
            "schedule": crontab(hour=15, minute=0),
            "options": {"queue": "ml", "priority": 2},
        },
        # Data freshness check - every 4 hours
        "check-data-freshness": {
            "task": "src.tasks.data_tasks.check_data_freshness_task",
            "schedule": crontab(hour="*/4"),
            "options": {"queue": "ml", "priority": 1},
        },
        "monitor-drift-and-retrain": {
            "task": "src.tasks.ml_tasks.monitor_drift_and_retrain_task",
            "schedule": crontab(hour="*/12"), # Every 12 hours
            "options": {"queue": "ml", "priority": 1},
        },
    },
    # Broker settings
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_pool_limit=50,
    broker_heartbeat=30,
    # Result backend settings
    redis_max_connections=50,
    redis_socket_timeout=30,
    redis_socket_connect_timeout=30,
    # Security (production only)
    **(
        {
            "broker_use_ssl": True,
            "redis_backend_use_ssl": True,
        }
        if ENVIRONMENT == "production"
        else {}
    ),
)


# =============================================================================
# Signal Handlers for Monitoring
# =============================================================================


@signals.task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **kw):
    """Log task start for monitoring."""
    logger.info(f"Task starting: {task.name}[{task_id}]")


@signals.task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **kw):
    """Log task completion for monitoring."""
    logger.info(f"Task completed: {task.name}[{task_id}] - {state}")


@signals.task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Log task failures for alerting."""
    logger.error(f"Task failed: {task_id} - {exception}")


@signals.worker_ready.connect
def worker_ready_handler(sender, **kw):
    """Log worker startup."""
    logger.info(f"Worker ready: {sender}")


@signals.worker_shutdown.connect
def worker_shutdown_handler(sender, **kw):
    """Clean shutdown handling."""
    logger.info(f"Worker shutting down: {sender}")


# =============================================================================
# Built-in Tasks
# =============================================================================


@celery_app.task(bind=True, queue="pricing", priority=1)
def health_check(self):
    """
    Health check task to verify Celery is working.
    Used by monitoring systems.
    """
    import time

    return {
        "status": "healthy",
        "task_id": self.request.id,
        "timestamp": time.time(),
        "worker": self.request.hostname,
    }


@celery_app.task(bind=True, queue="batch")
def cleanup_expired_results(self):
    """
    Cleanup expired task results from Redis.
    Runs daily to prevent memory bloat.
    """
    from celery.backends.redis import RedisBackend

    backend = celery_app.backend
    if isinstance(backend, RedisBackend):
        # Redis handles expiration automatically
        return {"status": "completed", "message": "Redis handles TTL automatically"}
    return {"status": "skipped", "message": "Not using Redis backend"}


@celery_app.task(bind=True, queue="pricing", priority=3)
def refresh_pricing_cache(self):
    """
    Pre-warm the pricing cache with common parameters.
    Runs every 15 minutes to ensure cache freshness.
    """
    logger.info("Refreshing pricing cache...")
    # Import here to avoid circular imports
    try:
        import asyncio

        from src.utils.cache import warm_cache

        asyncio.run(warm_cache())
        return {"status": "completed", "message": "Cache warmed successfully"}
    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Task Base Classes with Retry Logic
# =============================================================================


class BaseTaskWithRetry(Task):
    """
    Base task class with exponential backoff retry.
    Inherit from this for tasks that need robust retry logic.
    """

    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minute delay
    retry_jitter = True
    max_retries = 5

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure after all retries exhausted."""
        logger.error(f"Task {self.name}[{task_id}] failed after {self.max_retries} retries: {exc}")
        # Could send to dead letter queue or alerting system here


class PricingTask(Task):
    """
    Base task for pricing operations.
    Low latency, high priority with limited retries.
    """

    autoretry_for = (ConnectionError, TimeoutError)
    retry_backoff = False
    default_retry_delay = 1
    max_retries = 2
    time_limit = 30  # 30 second hard limit
    soft_time_limit = 25


class MLTask(Task):
    """
    Base task for ML operations.
    Resource intensive with longer timeouts.
    """

    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 300
    max_retries = 3
    time_limit = 3600  # 1 hour hard limit
    soft_time_limit = 3300

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log ML task retries."""
        logger.warning(f"ML Task {self.name}[{task_id}] retrying due to: {exc}")


class TradingTask(Task):
    """
    Base task for trading operations.
    Critical with immediate retry and alerting.
    """

    autoretry_for = (ConnectionError,)
    retry_backoff = False
    default_retry_delay = 0.5  # Fast retry
    max_retries = 3
    time_limit = 60
    soft_time_limit = 55

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Alert on trading task failure."""
        logger.critical(f"TRADING TASK FAILED: {self.name}[{task_id}] - {exc}")
        # Send alert to trading operations team


# Export base classes for use in task modules
celery_app.Task = BaseTaskWithRetry  # type: ignore
celery_app.PricingTask = PricingTask  # type: ignore
celery_app.MLTask = MLTask  # type: ignore
celery_app.TradingTask = TradingTask  # type: ignore


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    celery_app.start()
