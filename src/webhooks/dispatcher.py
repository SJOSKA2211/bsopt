import hmac
import hashlib
import time
import asyncio
import httpx
import structlog
from typing import Optional, Dict, Any, Union

logger = structlog.get_logger()

def _sign_payload(secret: str, timestamp: int, payload: str) -> str:
    """Helper to generate the HMAC-SHA256 signature."""
    signed_payload = f"{timestamp}.{payload}".encode('utf-8') # Ensure payload is bytes
    h = hmac.new(secret.encode('utf-8'), signed_payload, hashlib.sha256)
    return h.hexdigest()

async def _generate_signature(secret: str, payload: str, timestamp: Optional[int] = None) -> str:
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
from src.utils.cache import get_redis

# ... (HMAC functions remain)

class WebhookDispatcher:
    def __init__(self, celery_app: Any, circuit_breaker: Union[DistributedCircuitBreaker, InMemoryCircuitBreaker], dlq_task: Any):
        self.celery_app = celery_app
        self.circuit_breaker = circuit_breaker
        self.dlq_task = dlq_task # Celery task to send to DLQ

    async def _send_http_request(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        # Create a new client for each request to avoid RuntimeError "Cannot reopen a client instance"
        async with httpx.AsyncClient() as client:
            return await client.post(url, json=payload, headers=headers)

    async def dispatch_webhook(self, url: str, payload: Dict[str, Any], headers: Dict[str, str], secret: str, retries: int = 0):
        # We wrap the internal dispatch logic with the circuit breaker
        @self.circuit_breaker
        async def _dispatch():
            # Automatically sign the payload if a secret is provided and signature header is missing
            if secret and "X-Webhook-Signature" not in headers:
                import json
                # Use deterministic JSON serialization for consistent signatures
                payload_str = json.dumps(payload, sort_keys=True)
                signature_header = await _generate_signature(secret, payload_str)
                headers["X-Webhook-Signature"] = signature_header

            response = await self._send_http_request(url, payload, headers)
            response.raise_for_status() # Raise an exception for 4xx/5xx responses
            logger.info("webhook_dispatch_success", url=url, status_code=response.status_code)
            return response

        try:
            await _dispatch()
        except Exception as e:
            # Check if it was a circuit breaker rejection or a real failure
            error_str = str(e)
            if "Circuit Breaker" in error_str and "OPEN" in error_str:
                logger.warning("webhook_dispatch_skipped", url=url, reason="circuit_breaker_open")
                await self.dlq_task.delay({
                    "url": url, "payload": payload, "headers": headers, "secret": secret,
                    "reason": "circuit_breaker_open"
                })
                return

            logger.error("webhook_dispatch_failed", url=url, retries=retries, error=error_str)
            
            # Retry logic
            if retries < 5:
                retry_delay = 2 ** retries
                logger.info("webhook_dispatch_retrying", url=url, retry_delay=retry_delay, retries=retries + 1)
                await asyncio.sleep(retry_delay)
                await self.dispatch_webhook(url, payload, headers, secret, retries + 1)
            else:
                logger.error("webhook_dispatch_failed_max_retries", url=url)
                await self.dlq_task.delay({
                    "url": url, "payload": payload, "headers": headers, "secret": secret,
                    "reason": "max_retries_reached"
                })

