"""
Audit Logging Service
=====================

Provides structured logging for critical security and business events.
Supports both file logging and database persistence.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Request

from src.database import get_session
from src.database.models import AuditLog, User
from src.utils.sanitization import mask_email

# Use a dedicated logger for audit trails
audit_logger = logging.getLogger("audit")


class AuditEvent(str, Enum):
    """All auditable security events."""

    # Auth events
    USER_LOGIN_SUCCESS = "USER_LOGIN_SUCCESS"
    USER_LOGIN_FAILURE = "USER_LOGIN_FAILURE"
    USER_LOGOUT = "USER_LOGOUT"
    USER_REGISTER = "USER_REGISTER"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"

    # MFA events
    MFA_ENABLED = "MFA_ENABLED"
    MFA_VERIFIED = "MFA_VERIFIED"
    MFA_DISABLED = "MFA_DISABLED"
    MFA_LOGIN_SUCCESS = "MFA_LOGIN_SUCCESS"
    MFA_LOGIN_FAILURE = "MFA_LOGIN_FAILURE"

    # Password events
    PASSWORD_CHANGED = "PASSWORD_CHANGED"
    PASSWORD_RESET_REQUEST = "PASSWORD_RESET_REQUEST"
    PASSWORD_RESET_SUCCESS = "PASSWORD_RESET_SUCCESS"

    # Token events
    TOKEN_REFRESH = "TOKEN_REFRESH"
    TOKEN_REVOKED = "TOKEN_REVOKED"

    # Access events
    PROTECTED_RESOURCE_ACCESS = "PROTECTED_RESOURCE_ACCESS"
    UNAUTHORIZED_ACCESS_ATTEMPT = "UNAUTHORIZED_ACCESS_ATTEMPT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Security events
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    IP_BLOCKED = "IP_BLOCKED"


def log_audit(
    event: AuditEvent,
    user: Optional[User] = None,
    request: Optional[Request] = None,
    details: Optional[Dict[str, Any]] = None,
    persist_to_db: bool = True,
):
    """
    Log a structured audit event to both file and database.

    Args:
        event: The type of event that occurred.
        user: The user associated with the event.
        request: The FastAPI request object, if available.
        details: Additional structured details about the event.
        persist_to_db: Whether to save to database (default True).
    """
    log_data: Dict[str, Any] = {
        "event_type": event.value,
        "user_id": str(user.id) if user else None,
        "user_email": mask_email(user.email) if user else None,
    }

    source_ip = None
    user_agent = None
    request_path = None
    request_method = None

    if request:
        source_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        request_path = str(request.url.path)
        request_method = request.method

        log_data["source_ip"] = source_ip
        log_data["user_agent"] = user_agent
        log_data["request_path"] = request_path
        log_data["request_method"] = request_method

    if details:
        log_data["details"] = details

    # Log to file
    audit_logger.info(log_data)

    # Persist to database asynchronously via Celery task
    if persist_to_db:
        try:
            from src.tasks.audit_tasks import persist_audit_log # Import here to avoid circular dependency
            persist_audit_log.delay(
                event_type=event.value,
                user_id=str(user.id) if user else None,
                user_email=mask_email(user.email) if user else None,
                source_ip=source_ip,
                user_agent=user_agent[:500] if user_agent else None,
                request_path=request_path[:500] if request_path else None,
                request_method=request_method,
                details=details,
            )
        except Exception as e:
            # Log this error, but don't re-raise as the main request should not be blocked
            audit_logger.error(f"Failed to dispatch audit log to Celery task: {e}")
