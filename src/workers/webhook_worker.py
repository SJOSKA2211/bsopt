import asyncio
import os
from celery import Celery
from celery.exceptions import MaxRetriesExceededError # Import MaxRetriesExceededError
import structlog
from src.webhooks.dispatcher import WebhookDispatcher, CircuitBreaker
import httpx # Required for WebhookDispatcher

logger = structlog.get_logger()

celery_app = Celery("webhook_worker", broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"))

# Initialize dispatcher outside task to reuse connections/circuit breaker state
# In a real setup, this might be managed more dynamically or per worker process
# For simplicity, we initialize once.
_webhook_dispatcher = None

def get_webhook_dispatcher():
    global _webhook_dispatcher
    if _webhook_dispatcher is None:
        # Placeholder for real DLQ task. In production, this would be a separate Celery task.
        # For now, it's just a direct call for testing purposes or simple logging.
        class MockDlqTask:
            def delay(self, *args, **kwargs):
                logger.error("dlq_mock_called", args=args, kwargs=kwargs)
        
        circuit_breaker = CircuitBreaker()
        _webhook_dispatcher = WebhookDispatcher(
            celery_app=celery_app, # Pass celery_app for task retry handling
            circuit_breaker=circuit_breaker,
            dlq_task=send_to_dlq_task # Pass the actual DLQ task
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
            secret=secret, 
            retries=task_self.request.retries
        )
        logger.info("process_webhook_task_completed", url=url)
    except Exception as e:
        logger.error("process_webhook_task_failed", url=url, error=str(e), retries=task_self.request.retries)
        try:
            task_self.retry(exc=e)
        except MaxRetriesExceededError: # Use the imported exception
            logger.error("process_webhook_task_max_retries", url=url)
            send_to_dlq_task.delay(webhook_data, reason=f"celery_max_retries: {str(e)}")

@celery_app.task(bind=True, max_retries=5, default_retry_delay=1)
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