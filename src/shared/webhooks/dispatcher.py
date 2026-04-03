import hashlib
import hmac
import time
from typing import Any

import orjson
import structlog

from src.shared.utils.circuit_breaker import DistributedCircuitBreaker, InMemoryCircuitBreaker
from src.shared.utils.http_client import HttpClientManager

logger = structlog.get_logger(__name__)


def sign_payload(secret: str, timestamp: int, payload: str) -> str:
    """Helper to generate the HMAC-SHA256 signature."""
    signed_payload = f"{timestamp}.{payload}".encode()
    h = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256)
    return h.hexdigest()


async def generate_signature(secret: str, payload: str, timestamp: int | None = None) -> str:
    """
    Generates a Stripe-style HMAC-SHA256 signature for a webhook payload.
    Format: t=<timestamp>,sha256=<signature>
    """
    if timestamp is None:
        timestamp = int(time.time())

    signature = sign_payload(secret, timestamp, payload)
    return f"t={timestamp},sha256={signature}"


async def verify_signature(
    secret: str, payload: str, timestamp: int, signature: str, tolerance: int = 300
) -> bool:
    """
    Verifies a Stripe-style HMAC-SHA256 webhook signature.
    """
    now = int(time.time())
    if abs(now - timestamp) > tolerance:
        logger.warning(
            "webhook_signature_timestamp_mismatch",
            timestamp=timestamp,
            now=now,
            tolerance=tolerance,
        )
        return False

    expected_signature = sign_payload(secret, timestamp, payload)
    return hmac.compare_digest(expected_signature, signature)


class WebhookDispatcher:
    """
    Centralized dispatcher for outgoing webhooks.
    Handles serialization, signing, and circuit breaking.
    """

    def __init__(
        self,
        circuit_breaker: DistributedCircuitBreaker | InMemoryCircuitBreaker,
        celery_app: Any = None,
        dlq_task: Any = None,
    ):
        self.circuit_breaker = circuit_breaker
        self.celery_app = celery_app
        self.dlq_task = dlq_task

    async def dispatch_webhook(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        secret: str | None = None,
        retries: int = 0,
    ) -> Any:
        """
        Dispatches a webhook payload to the specified URL.
        """
        headers = headers or {}

        @self.circuit_breaker
        async def _dispatch():
            # Serialize once for consistency
            payload_bytes = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
            payload_str = payload_bytes.decode("utf-8")

            if secret and "X-Webhook-Signature" not in headers:
                signature_header = await generate_signature(secret, payload_str)
                headers["X-Webhook-Signature"] = signature_header

            client = HttpClientManager.get_client()
            response = await client.post(
                url,
                content=payload_bytes,
                headers={**headers, "Content-Type": "application/json"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response

        try:
            return await _dispatch()
        except Exception as e:
            error_str = str(e)
            if "Circuit Breaker" in error_str and "OPEN" in error_str:
                logger.warning("webhook_dispatch_skipped_cb_open", url=url)
                raise

            logger.error("webhook_dispatch_failed", url=url, error=error_str, retries=retries)
            raise
