import asyncio
import os

import structlog
from celery import Celery
from celery.exceptions import MaxRetriesExceededError

from src.shared.webhooks.dispatcher import WebhookDispatcher

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = structlog.get_logger(__name__)

celery_app = Celery(
    "webhook_worker", broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
)

_webhook_dispatcher = None


def get_webhook_dispatcher():
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        from src.shared.utils.circuit_breaker import (
            DistributedCircuitBreaker,
            InMemoryCircuitBreaker,
        )

        # This is a bit synchronous for a getter, but in Celery workers it runs in an init hook or first task
        # We'll use a simplified check for circuit breaker
        try:
            # We don't want to block indefinitely here
            circuit_breaker = DistributedCircuitBreaker(
                name="webhook_dispatch",
                redis_client=None,  # Will be set on first use or use fallback
                failure_threshold=5,
                recovery_timeout=30,
            )
        except Exception:
            circuit_breaker = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30)

        _webhook_dispatcher = WebhookDispatcher(
            circuit_breaker=circuit_breaker, celery_app=celery_app, dlq_task=send_to_dlq_task
        )
    return _webhook_dispatcher


async def _process_webhook_core(task_self, webhook_data: dict):
    dispatcher = get_webhook_dispatcher()
    url = webhook_data["url"]
    payload = webhook_data["payload"]
    headers = webhook_data.get("headers", {})
    secret = webhook_data.get("secret")
    retries = task_self.request.retries

    try:
        await dispatcher.dispatch_webhook(
            url=url, payload=payload, headers=headers, secret=secret, retries=retries
        )
        logger.info("webhook_worker_success", url=url)
    except Exception as e:
        error_str = str(e)
        if "Circuit Breaker" in error_str and "OPEN" in error_str:
            logger.warning("webhook_worker_cb_open", url=url)
            raise task_self.retry(exc=e, countdown=60)

        logger.error("webhook_worker_failed", url=url, error=error_str, retry=retries)
        try:
            retry_delay = 2**retries
            raise task_self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            logger.error("webhook_worker_max_retries", url=url)
            send_to_dlq_task.delay(webhook_data, reason=f"max_retries: {error_str}")


@celery_app.task(bind=True, max_retries=5)
def process_webhook_task(self, webhook_data: dict):
    return asyncio.run(_process_webhook_core(self, webhook_data))


@celery_app.task
def send_to_dlq_task(webhook_data: dict, reason: str = "unknown"):
    """Persists failed webhooks to Redis DLQ."""
    logger.error("webhook_dlq_entry", url=webhook_data.get("url"), reason=reason)
    try:
        import json

        from src.shared.utils.cache import get_redis_client

        async def _persist():
            redis = await get_redis_client()
            await redis.lpush("webhook:dlq", json.dumps({"reason": reason, "data": webhook_data}))

        asyncio.run(_persist())
    except Exception as e:
        logger.error("webhook_dlq_persist_failed", error=str(e))
