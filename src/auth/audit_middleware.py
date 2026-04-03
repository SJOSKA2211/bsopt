import base64
import time
from typing import Any

import structlog
from fastapi import BackgroundTasks, Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.shared.config import settings
from src.shared.rabbitmq import get_rabbitmq
from src.shared.utils.crypto import AES256GCM

logger = structlog.get_logger(__name__)

# Initialize Audit Vault
_vault_key = settings.AUDIT_VAULT_KEY
_vault = (
    AES256GCM(base64.urlsafe_b64encode(_vault_key.encode()[:32]).decode()) if _vault_key else None
)


async def _produce_audit_log(payload: dict[str, Any]):
    """Background task to produce audit logs to RabbitMQ with delivery assurance."""
    try:
        # OPTIMIZATION: Move encryption to background task to minimize request latency
        if _vault:
            if "user_email" in payload and payload["user_email"] not in ("anonymous", None):
                payload["user_email"] = _vault.encrypt(str(payload["user_email"]).encode())
            if "client_ip" in payload and payload["client_ip"] not in ("unknown", None):
                payload["client_ip"] = _vault.encrypt(str(payload["client_ip"]).encode())
            if "user_id" in payload and payload["user_id"] not in ("anonymous", None):
                # Only encrypt if it's a real user_id
                payload["user_id"] = _vault.encrypt(str(payload["user_id"]).encode())

        rmq = get_rabbitmq()
        await rmq.publish_audit(payload)
        logger.debug("audit_log_published_to_rabbitmq")
    except Exception as e:
        logger.warning("audit_log_production_failed", error=str(e))


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Production Audit Middleware.
    Pushes audit logs to RabbitMQ asynchronously.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Capture audit information (fast path)
        user_id = getattr(request.state, "user_id", "anonymous")
        user_email = getattr(request.state, "user_email", "anonymous")

        audit_payload = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "user_id": user_id,
            "user_email": user_email,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "latency_ms": (time.time() - start_time) * 1000,
        }

        # Use BackgroundTasks to offload the production
        if "background_tasks" not in request.scope:
            request.scope["background_tasks"] = BackgroundTasks()

        request.scope["background_tasks"].add_task(_produce_audit_log, audit_payload)

        return response
