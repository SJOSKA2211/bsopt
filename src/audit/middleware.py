import base64
import time
from typing import Any

import orjson
import structlog
from fastapi import BackgroundTasks, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

import os

from src.utils.crypto import AES256GCM

# 🚀 SINGULARITY: Initialize Audit Vault
_vault_key = os.getenv("AUDIT_VAULT_KEY", "changeme_32byte_key_for_god_mode!")
_vault = (
    AES256GCM(base64.urlsafe_b64encode(_vault_key.encode()[:32]).decode())
    if _vault_key
    else None
)


def _produce_audit_log(producer: Any, topic: str, payload: dict[str, Any]):
    """Background task to produce audit logs to Kafka with delivery assurance."""
    try:
        # 🚀 OPTIMIZATION: Encrypt PII fields at rest in Kafka/Loki
        if _vault:
            if "user_id" in payload:
                payload["user_id"] = _vault.encrypt(str(payload["user_id"]).encode())
            if "client_ip" in payload:
                payload["client_ip"] = _vault.encrypt(
                    str(payload["client_ip"]).encode()
                )

        producer.produce(
            topic,
            orjson.dumps(payload),
            on_delivery=lambda err, msg: (
                logger.debug("audit_log_delivered", topic=msg.topic())
                if not err
                else logger.error("audit_log_delivery_failed", error=str(err))
            ),
        )
        # SOTA: Poll with a small timeout to allow for network batching
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
            # Capture audit information
            user = getattr(request.state, "user", {})
            if hasattr(user, "id"):
                user_id = str(user.id)
            elif isinstance(user, dict):
                user_id = user.get("sub", "anonymous")
            else:
                user_id = "anonymous"

            audit_payload = {
                "timestamp": time.time(),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "user_id": user_id,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "latency_ms": (time.time() - start_time) * 1000,
            }

            # Use BackgroundTasks to offload the production to avoid blocking the API thread
            if "background_tasks" not in request.scope:
                request.scope["background_tasks"] = BackgroundTasks()

            request.scope["background_tasks"].add_task(
                _produce_audit_log, producer, self.topic, audit_payload
            )

        return response
