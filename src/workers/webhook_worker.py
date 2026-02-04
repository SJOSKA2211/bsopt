import asyncio
import os
from celery import Celery
from celery.exceptions import MaxRetriesExceededError # Import MaxRetriesExceededError
import structlog
from src.webhooks.dispatcher import WebhookDispatcher, CircuitBreaker
import httpx # Required for WebhookDispatcher

# Optimized event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = structlog.get_logger()

celery_app = Celery("webhook_worker", broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"))

# Initialize dispatcher outside task to reuse connections/circuit breaker state
# In a real setup, this might be managed more dynamically or per worker process
# For simplicity, we initialize once.
_webhook_dispatcher = None

def get_webhook_dispatcher():
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        from src.utils.cache import get_redis
        from src.utils.circuit_breaker import DistributedCircuitBreaker
        
        redis_client = get_redis()
        if redis_client is None:
            # Fallback for tests or local dev without redis
            from src.utils.circuit_breaker import InMemoryCircuitBreaker
            logger.warning("webhook_dispatcher_fallback_in_memory")
            circuit_breaker = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        else:
            circuit_breaker = DistributedCircuitBreaker(
                name="webhook_dispatch",
                redis_client=redis_client,
                failure_threshold=5,
                recovery_timeout=30
            )
            
        _webhook_dispatcher = WebhookDispatcher(
            celery_app=celery_app, 
            circuit_breaker=circuit_breaker,
            dlq_task=send_to_dlq_task
        )
    return _webhook_dispatcher

async def _process_webhook_core(task_self, webhook_data: dict):
    dispatcher = get_webhook_dispatcher()
    url = webhook_data["url"]
    payload = webhook_data["payload"]
    headers = webhook_data["headers"]
    secret = webhook_data["secret"]
    
    try:
        await dispatcher.dispatch_webhook(
            url=url, 
            payload=payload, 
            headers=headers, 
            secret=secret
        )
        logger.info("process_webhook_task_completed", url=url)
    except Exception as e:
        error_str = str(e)
        if "Circuit Breaker" in error_str and "OPEN" in error_str:
            # For circuit breaker, retry with a longer delay or send to DLQ
            logger.warning("webhook_worker_circuit_breaker_open", url=url)
            raise task_self.retry(exc=e, countdown=60) # Long delay
            
        logger.error("process_webhook_task_failed", url=url, error=error_str, retries=task_self.request.retries)
        try:
            # Use exponential backoff for retries
            retry_delay = 2 ** task_self.request.retries
            raise task_self.retry(exc=e, countdown=retry_delay)
        except MaxRetriesExceededError:
            logger.error("process_webhook_task_max_retries", url=url)
            send_to_dlq_task.delay(webhook_data, reason=f"celery_max_retries: {error_str}")

@celery_app.task(bind=True, max_retries=5)
def process_webhook_task(self, webhook_data: dict):
    asyncio.run(_process_webhook_core(self, webhook_data))

@celery_app.task
def send_to_dlq_task(webhook_data: dict, reason: str = "unknown_failure"):
    """
    Task to handle webhooks that failed after all retries or due to circuit breaker.
    """
    logger.error("webhook_sent_to_dlq", url=webhook_data.get("url"), reason=reason, webhook_data=webhook_data)
    # In a real system, this would store the webhook in a persistent DLQ
    # for manual inspection or re-processing later.