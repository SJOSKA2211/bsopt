import structlog
from typing import Any, Dict, Optional

from src.tasks.celery_app import celery_app
from src.database import get_session
from src.database.models import AuditLog

logger = structlog.get_logger(__name__)

@celery_app.task(name="audit_tasks.persist_audit_log", acks_late=True)
def persist_audit_log(
    event_type: str,
    user_id: Optional[str],
    user_email: Optional[str], # Already masked
    source_ip: Optional[str],
    user_agent: Optional[str],
    request_path: Optional[str],
    request_method: Optional[str],
    details: Optional[Dict[str, Any]],
) -> None:
    """
    Celery task to asynchronously persist an audit log to the database.
    """
    session = None
    try:
        session = get_session()
        audit_log = AuditLog(
            event_type=event_type,
            user_id=user_id,
            user_email=user_email,
            source_ip=source_ip,
            user_agent=user_agent[:500] if user_agent else None,
            request_path=request_path[:500] if request_path else None,
            request_method=request_method,
            details=details,
        )
        session.add(audit_log)
        session.commit()
        logger.info(f"Audit log persisted: {event_type} for user {user_id}")
    except Exception as e:
        if session:
            session.rollback()
        logger.error(f"Failed to persist audit log to database via Celery task: {e}")
    finally:
        if session:
            session.close()
