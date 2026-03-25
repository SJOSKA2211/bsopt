import time
from uuid import uuid4

import structlog
from celery import Task
from sqlalchemy import update

from src.database import get_async_db_context
from src.database.models import EmailLog
from src.shared.config import settings
from src.shared.email import TransactionalEmailService
from src.workers.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

# Initialize email service
email_service = TransactionalEmailService(
    api_key=settings.SENDGRID_API_KEY,
    from_email=settings.FROM_EMAIL
)

class EmailAuditTask(Task):
    """
    Base task for emails that provides automatic auditing and retries.
    """
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("email_task_failed", task_id=task_id, error=str(exc))

@celery_app.task(bind=True, base=EmailAuditTask, max_retries=3)
def send_transactional_email(self, to_email: str, subject: str, template_name: str, context: dict):
    """
    Sends a transactional email and logs the attempt to the database for auditing.
    """
    log_id = uuid4()
    start_time = time.time()

    # 1. Pre-log (Transactional Audit Hook)
    async def _create_log():
        async with get_async_db_context() as db:
            log = EmailLog(
                id=log_id,
                recipient=to_email,
                subject=subject,
                template_name=template_name,
                status="pending"
            )
            db.add(log)
            await db.commit()

    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(_create_log())

    # 2. Dispatch
    try:
        success = email_service.send_single_email(
            to_email=to_email,
            subject=subject,
            template_name=template_name,
            context=context
        )
        
        status = "sent" if success else "failed"
        duration_ms = (time.time() - start_time) * 1000

        # 3. Finalize log
        async def _finalize_log():
            async with get_async_db_context() as db:
                await db.execute(
                    update(EmailLog)
                    .where(EmailLog.id == log_id)
                    .values(
                        status=status,
                        sent_at=asyncio.get_event_loop().time() if success else None, # Simplified
                        duration_ms=duration_ms
                    )
                )
                await db.commit()

        loop.run_until_complete(_finalize_log())
        
        if not success:
            raise self.retry(countdown=60)
            
        return {"status": "sent", "log_id": str(log_id)}

    except Exception as e:
        logger.error("email_dispatch_error", error=str(e), to_email=to_email)
        
        async def _log_error():
            async with get_async_db_context() as db:
                await db.execute(
                    update(EmailLog)
                    .where(EmailLog.id == log_id)
                    .values(status="failed", error_message=str(e))
                )
                await db.commit()
        
        loop.run_until_complete(_log_error())
        raise self.retry(exc=e, countdown=120)

@celery_app.task(base=EmailAuditTask)
def send_batch_marketing_emails(recipients: list, subject: str, template_name: str):
    """
    Sends batch emails (Auditing simplified for batches).
    """
    return email_service.send_batch_emails(recipients, subject, template_name)
