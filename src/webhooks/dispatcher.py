import hmac
import hashlib
import time
import asyncio
import httpx
import structlog
from typing import Optional, Dict, Any

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

# Circuit Breaker States
CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30, expected_successes: int = 1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_successes = expected_successes
        
        self._state = CLOSED
        self._failures = 0
        self._successes = 0
        self._last_failure_time = 0
        
    @property
    def is_closed(self):
        return self._state == CLOSED

    @property
    def is_open(self):
        if self._state == OPEN and (time.time() - self._last_failure_time > self.recovery_timeout):
            self._state = HALF_OPEN
        return self._state == OPEN
    
    @property
    def is_half_open(self):
        # Accessing is_open will trigger the state transition to HALF_OPEN if conditions met
        _ = self.is_open 
        return self._state == HALF_OPEN

    async def record_failure(self):
        if self._state == HALF_OPEN:
            self._state = OPEN
            self._last_failure_time = time.time()
        else:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = OPEN
                self._last_failure_time = time.time()
        logger.warning("circuit_breaker_failure", state=self._state, failures=self._failures)

    async def record_success(self):
        if self._state == HALF_OPEN:
            self._successes += 1
            if self._successes >= self.expected_successes:
                self._state = CLOSED
                self._failures = 0
                self._successes = 0
            else: # Not enough successes, remain half-open or go back to open
                self._state = OPEN
                self._last_failure_time = time.time()
        elif self._state == CLOSED:
            self._failures = 0 # Reset failures on success in closed state
        logger.info("circuit_breaker_success", state=self._state)

class WebhookDispatcher:
    def __init__(self, celery_app: Any, circuit_breaker: CircuitBreaker, dlq_task: Any):
        self.celery_app = celery_app
        self.circuit_breaker = circuit_breaker
        self.dlq_task = dlq_task # Celery task to send to DLQ
        # self.http_client = httpx.AsyncClient() # Re-use HTTP client - no longer class member

    async def _send_http_request(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        # Create a new client for each request to avoid RuntimeError "Cannot reopen a client instance"
        async with httpx.AsyncClient() as client:
            return await client.post(url, json=payload, headers=headers)

    async def dispatch_webhook(self, url: str, payload: Dict[str, Any], headers: Dict[str, str], secret: str, retries: int = 0):
        if self.circuit_breaker.is_open:
            logger.warning("webhook_dispatch_skipped", url=url, reason="circuit_breaker_open")
            # Immediately send to DLQ if circuit breaker is open, no retries
            await self.dlq_task.delay({
                "url": url, "payload": payload, "headers": headers, "secret": secret,
                "reason": "circuit_breaker_open"
            })
            return

        try:
            response = await self._send_http_request(url, payload, headers)
            response.raise_for_status() # Raise an exception for 4xx/5xx responses
            await self.circuit_breaker.record_success()
            logger.info("webhook_dispatch_success", url=url, status_code=response.status_code)
        except httpx.HTTPStatusError as e:
            await self.circuit_breaker.record_failure()
            logger.error("webhook_dispatch_http_error", url=url, status_code=e.response.status_code, retries=retries, error=str(e))
            if retries < 5: # Max 5 retries with exponential backoff
                retry_delay = 2 ** retries # Exponential backoff
                logger.info("webhook_dispatch_retrying", url=url, retry_delay=retry_delay, retries=retries + 1)
                await asyncio.sleep(retry_delay)
                await self.dispatch_webhook(url, payload, headers, secret, retries + 1)
            else:
                logger.error("webhook_dispatch_failed_retries", url=url, retries=retries)
                await self.dlq_task.delay({
                    "url": url, "payload": payload, "headers": headers, "secret": secret,
                    "reason": "max_retries_reached"
                })
        except httpx.RequestError as e:
            await self.circuit_breaker.record_failure()
            logger.error("webhook_dispatch_request_error", url=url, retries=retries, error=str(e))
            if retries < 5: # Max 5 retries with exponential backoff
                retry_delay = 2 ** retries # Exponential backoff
                logger.info("webhook_dispatch_retrying", url=url, retry_delay=retry_delay, retries=retries + 1)
                await asyncio.sleep(retry_delay)
                await self.dispatch_webhook(url, payload, headers, secret, retries + 1)
            else:
                logger.error("webhook_dispatch_failed_retries", url=url, retries=retries)
                await self.dlq_task.delay({
                    "url": url, "payload": payload, "headers": headers, "secret": secret,
                    "reason": "max_retries_reached"
                })
        except Exception as e:
            await self.circuit_breaker.record_failure()
            logger.error("webhook_dispatch_unexpected_error", url=url, retries=retries, error=str(e))
            await self.dlq_task.delay({
                "url": url, "payload": payload, "headers": headers, "secret": secret,
                "reason": f"unexpected_error: {str(e)}"
            })

