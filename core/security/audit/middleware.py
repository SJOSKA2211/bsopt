import base64
import time
from typing import Any

import msgspec
import structlog
from fastapi import BackgroundTasks, Request
from starlette.middleware.base import BaseHTTPMiddleware

from core.shared.config import settings
from core.shared.utils.crypto import AES256GCM

logger = structlog.get_logger(__name__)

# Initialize Audit Vault
_vault_key = settings.AUDIT_VAULT_KEY
_vault = (
    AES256GCM(base64.urlsafe_b64encode(_vault_key.encode()[:32]).decode()) if _vault_key else None
)


def _produce_audit_log(producer: Any, topic: str, payload: dict[str, Any]):
    """Background task to produce audit logs to Kafka with delivery assurance."""
    try:
        #  OPTIMIZATION: Encrypt PII fields at rest in Kafka/Loki
        if _vault:
            if "user_id" in payload:
                payload["user_id"] = _vault.encrypt(str(payload["user_id"]).encode())
            if "client_ip" in payload:
                payload["client_ip"] = _vault.encrypt(str(payload["client_ip"]).encode())

        producer.produce(
            topic,
            msgspec.json.encode(payload),
            on_delivery=lambda err, msg: (
                logger.debug("audit_log_delivered", topic=msg.topic())
                if not err
                else logger.error("audit_log_delivery_failed", error=str(err))
            ),
        )
        # OPTIMIZED: Poll with a small timeout to allow for network batching
        producer.poll(0.1)
    except Exception as e:
        logger.warning("audit_log_production_failed", error=str(e))


def flush_audit_producer(producer: Any):
    """Explicitly flush pending audit logs during shutdown."""
    if producer:
        logger.info("flushing_audit_logs")
        producer.flush(10.0)  # 10s timeout


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, producer: Any = None, topic: str = "audit-logs"):
        super().__init__(app)
        self.producer = producer
        self.topic = topic

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Get producer from app state if not provided at init
        producer = self.producer or getattr(request.app.state, "audit_producer", None)

        if producer:
            # Capture audit information (Updated for consolidated request.state)
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

            if _vault:
                # Encrypt sensitive fields
                if "user_email" in audit_payload and audit_payload["user_email"] != "anonymous":
                    audit_payload["user_email"] = _vault.encrypt(
                        str(audit_payload["user_email"]).encode()
                    )
                if "client_ip" in audit_payload and audit_payload["client_ip"] != "unknown":
                    audit_payload["client_ip"] = _vault.encrypt(
                        str(audit_payload["client_ip"]).encode()
                    )

            # Use BackgroundTasks to offload the production to avoid blocking the API thread
            if "background_tasks" not in request.scope:
                request.scope["background_tasks"] = BackgroundTasks()

            request.scope["background_tasks"].add_task(
                _produce_audit_log, producer, self.topic, audit_payload
            )

        return response
