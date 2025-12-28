import asyncio
import logging
import os

from src.services.email_service import TransactionalEmailService
from src.utils.cache import RateLimitTier, rate_limiter

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Initialize Email Service
email_service = TransactionalEmailService(
    api_key=os.getenv("EMAIL_SERVICE_API_KEY", ""),
    from_email=os.getenv("FROM_EMAIL", "no-reply@bsopt.com"),
)


@celery_app.task(
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    name="src.tasks.email_tasks.send_transactional_email",
)
def send_transactional_email(self, to_email: str, subject: str, template_name: str, context: dict):
    """
    Async task to send a single transactional email with rate limiting.
    """
    # Check rate limit (system-wide for emails)
    # Note: In Celery worker, we use a different approach for async if needed,
    # but here we just run the coroutine.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    rl_result = loop.run_until_complete(
        rate_limiter.check_rate_limit(
            user_id="system_email", endpoint="send_email", tier=RateLimitTier.ENTERPRISE
        )
    )

    if not rl_result:  # Rate limit reached
        logger.warning("Email rate limit reached. Retrying in 60s...")
        raise self.retry(countdown=60)

    logger.info(f"Sending email to {to_email}")
    success = email_service.send_single_email(
        to_email=to_email, subject=subject, template_name=template_name, context=context
    )
    if not success:
        raise self.retry(exc=Exception("Failed to send email via SendGrid"))
    return {"status": "sent", "to": to_email}


@celery_app.task(name="src.tasks.email_tasks.send_batch_marketing_emails")
def send_batch_marketing_emails(recipients: list, subject: str, template_name: str):
    """
    Async task to send batch emails.
    """
    logger.info(f"Sending batch email to {len(recipients)} recipients")
    BATCH_SIZE = 100
    for i in range(0, len(recipients), BATCH_SIZE):
        batch = recipients[i : i + BATCH_SIZE]
        email_service.send_batch_emails(batch, subject, template_name)

    return {"status": "batch_sent", "count": len(recipients)}
