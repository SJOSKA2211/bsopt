import asyncio

import structlog

from src.workers.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

@celery_app.task
def persist_audit_log(
    event_type: str,
    user_id: str = None,
    user_email: str = None,
    source_ip: str = None,
    user_agent: str = None,
    request_path: str = None,
    request_method: str = None,
    details: dict = None,
):
    """
    Asynchronously persists an audit log entry to the database.
    """
    from src.database import get_async_db_context
    from src.database.models import AuditLog

    async def _persist():
        try:
            async with get_async_db_context() as db:
                log = AuditLog(
                    method=request_method or event_type[:10],
                    path=request_path or "",
                    status_code=200,
                    user_id=user_id,
                    client_ip=source_ip or "0.0.0.0",
                    user_agent=user_agent or "",
                    latency_ms=0.0,
                    details={
                        "event_type": event_type,
                        "user_email": user_email,
                        **(details or {}),
                    },
                )
                db.add(log)
                await db.commit()
        except Exception as e:
            logger.error("audit_log_persistence_failed", error=str(e))

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_persist())
        else:
            loop.run_until_complete(_persist())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_persist())
