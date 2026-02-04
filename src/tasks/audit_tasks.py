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
            method=request_method or "UNKNOWN",
            path=request_path[:500] if request_path else "UNKNOWN",
            status_code=0, # Default for async tasks
            user_id=user_id or "ANONYMOUS",
            client_ip=source_ip or "0.0.0.0",
            user_agent=user_agent[:500] if user_agent else "UNKNOWN",
            latency_ms=0.0,
            metadata_json={
                "event_type": event_type,
                "user_email": user_email,
                "details": details
            }
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
