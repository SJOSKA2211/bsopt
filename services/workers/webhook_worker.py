import asyncio
import sys
import time

import structlog
from celery import Celery
from celery.exceptions import MaxRetriesExceededError  # Import MaxRetriesExceededError

from services.config import settings
from core.shared.utils.celery import BaseAsyncTask
from core.shared.utils.lazy_import import lazy_import
from services.webhooks.dispatcher import WebhookDispatcher

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = structlog.get_logger()

celery_app = Celery("webhook_worker", broker=settings.broker_url)

# High-Availability Queue Configuration
celery_app.conf.task_queues = {
    "webhooks": {
        "exchange": "webhooks",
        "routing_key": "webhooks.#",
        "queue_arguments": {"x-queue-type": "quorum"}, # HA Quorum Queues (RabbitMQ 3.8+)
    }
}
celery_app.conf.task_default_queue = "webhooks"

# Initialize dispatcher outside task to reuse connections/circuit breaker state
# In a real setup, this might be managed more dynamically or per worker process
# For simplicity, we initialize once.


# Lazy Import Map
_IMPORT_MAP = {
    "get_redis": "core.shared.cache.get_redis",
    "DistributedCircuitBreaker": "core.shared.circuit_breaker.DistributedCircuitBreaker",
    "InMemoryCircuitBreaker": "core.shared.circuit_breaker.InMemoryCircuitBreaker",
}


def _get_attr(name: str):
    return lazy_import(__name__, _IMPORT_MAP, name, sys.modules[__name__])


_webhook_dispatcher = None


def get_webhook_dispatcher():
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        get_redis = _get_attr("get_redis")
        redis_client = get_redis()

        if redis_client is None:
            InMemoryCircuitBreaker = _get_attr("InMemoryCircuitBreaker")
            circuit_breaker = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        else:
            DistributedCircuitBreaker = _get_attr("DistributedCircuitBreaker")
            circuit_breaker = DistributedCircuitBreaker(
                name="webhook_dispatch", redis_client=redis_client
            )

        _webhook_dispatcher = WebhookDispatcher(
            celery_app=celery_app,
            circuit_breaker=circuit_breaker,
            dlq_task=send_to_dlq_task,
        )
    return _webhook_dispatcher


async def _process_webhook_core(task_self, webhook_data: dict):
    dispatcher = get_webhook_dispatcher()
    url = webhook_data["url"]
    payload = webhook_data["payload"]
    headers = webhook_data["headers"]
    secret = webhook_data["secret"]

    try:
        await dispatcher.dispatch_webhook(url=url, payload=payload, headers=headers, secret=secret)
        logger.info("process_webhook_task_completed", url=url)
    except Exception as e:
        error_str = str(e)
        if "Circuit Breaker" in error_str and "OPEN" in error_str:
            # For circuit breaker, retry with a longer delay or send to DLQ
            logger.warning("webhook_worker_circuit_breaker_open", url=url)
            raise task_self.retry(exc=e, countdown=60) from e  # Long delay

        logger.error(
            "process_webhook_task_failed",
            url=url,
            error=error_str,
            retries=task_self.request.retries,
        )
        try:
            # Use exponential backoff for retries
            retry_delay = 2**task_self.request.retries
            raise task_self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            logger.error("process_webhook_task_max_retries", url=url)
            send_to_dlq_task.delay(webhook_data, reason=f"celery_max_retries: {error_str}")


@celery_app.task(base=BaseAsyncTask, bind=True, max_retries=5)
def process_webhook_task(self, webhook_data: dict):
    return self.run_async(_process_webhook_core(self, webhook_data))


@celery_app.task
def send_to_dlq_task(webhook_data: dict, reason: str = "unknown_failure"):
    """
    Task to handle webhooks that failed after all retries or due to circuit breaker.
    """
    logger.error(
        "webhook_sent_to_dlq",
        url=webhook_data.get("url"),
        reason=reason,
        webhook_data=webhook_data,
    )
    
    # PERMANENT DLQ: Store in Redis for manual retry/inspection
    try:
        get_redis = _get_attr("get_redis")
        redis_client = get_redis()
        if redis_client:
            import json
            dlq_entry = {
                "data": webhook_data,
                "reason": reason,
                "timestamp": time.time()
            }
            # Push to the 'webhooks:dlq' list
            # Note: Using run_async if we want to be consistent, 
            # but this is a standard celery task (sync wrapper)
            return asyncio.run(redis_client.lpush("webhooks:dlq", json.dumps(dlq_entry)))
    except Exception as e:
        logger.critical("dlq_storage_failed", error=str(e), url=webhook_data.get("url"))
