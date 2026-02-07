import hashlib
import hmac
import time
from typing import Any

import structlog

logger = structlog.get_logger()

def _sign_payload(secret: str, timestamp: int, payload: str) -> str:
    """Helper to generate the HMAC-SHA256 signature."""
    signed_payload = f"{timestamp}.{payload}".encode() # Ensure payload is bytes
    h = hmac.new(secret.encode('utf-8'), signed_payload, hashlib.sha256)
    return h.hexdigest()

async def _generate_signature(secret: str, payload: str, timestamp: int | None = None) -> str:
    """
    Generates a Stripe-style HMAC-SHA256 signature for a webhook payload.
    Format: t=<timestamp>,sha256=<signature>
    """
    if timestamp is None:
        timestamp = int(time.time())
    
    signature = _sign_payload(secret, timestamp, payload)
    return f"t={timestamp},sha256={signature}"

async def _verify_signature(secret: str, payload: str, timestamp: int, signature: str, tolerance: int = 300) -> bool:
    """
    Verifies a Stripe-style HMAC-SHA256 webhook signature.
    :param secret: The webhook secret.
    :param payload: The raw request body.
    :param timestamp: The timestamp from the 't=' part of the signature header.
    :param signature: The signature part (after 'sha256=').
    :param tolerance: Time tolerance in seconds (default 5 minutes).
    :return: True if the signature is valid, False otherwise.
    """
    # 1. Check timestamp
    now = int(time.time())
    if abs(now - timestamp) > tolerance:
        logger.warning("webhook_signature_timestamp_mismatch", timestamp=timestamp, now=now, tolerance=tolerance)
        return False # Timestamp outside of tolerance

    # 2. Re-generate expected signature
    expected_signature = _sign_payload(secret, timestamp, payload)
    
    # 3. Compare signatures (constant-time comparison to prevent timing attacks)
    return hmac.compare_digest(expected_signature, signature)

from src.utils.circuit_breaker import DistributedCircuitBreaker, InMemoryCircuitBreaker
from src.utils.http_client import HttpClientManager


class WebhookDispatcher:
    def __init__(self, celery_app: Any, circuit_breaker: DistributedCircuitBreaker | InMemoryCircuitBreaker, dlq_task: Any):
        self.celery_app = celery_app
        self.circuit_breaker = circuit_breaker
        self.dlq_task = dlq_task # Celery task to send to DLQ

    async def dispatch_webhook(self, url: str, payload: dict[str, Any], headers: dict[str, str], secret: str):
        # We wrap the internal dispatch logic with the circuit breaker
        @self.circuit_breaker
        async def _dispatch():
            # Automatically sign the payload if a secret is provided and signature header is missing
            if secret and "X-Webhook-Signature" not in headers:
                import orjson
                # Use deterministic JSON serialization for consistent signatures
                payload_str = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode('utf-8')
                signature_header = await _generate_signature(secret, payload_str)
                headers["X-Webhook-Signature"] = signature_header

            client = HttpClientManager.get_client()
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status() # Raise an exception for 4xx/5xx responses
            logger.info("webhook_dispatch_success", url=url, status_code=response.status_code)
            return response

        try:
            await _dispatch()
        except Exception as e:
            # Check if it was a circuit breaker rejection
            error_str = str(e)
            if "Circuit Breaker" in error_str and "OPEN" in error_str:
                logger.warning("webhook_dispatch_skipped", url=url, reason="circuit_breaker_open")
                raise # Re-raise to let Celery handle retry/DLQ
            
            logger.error("webhook_dispatch_failed", url=url, error=error_str)
            raise # Re-raise for Celery retry

