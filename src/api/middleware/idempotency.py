import hashlib
import time

import msgspec
import redis.asyncio as redis
import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger()

# Metrics
IDEMPOTENCY_REQUESTS_TOTAL = Counter(
    "idempotency_requests_total", "Total idempotency requests", ["status"]
)
IDEMPOTENCY_CACHE_HITS = Counter(
    "idempotency_cache_hits_total", "Total idempotency cache hits"
)
IDEMPOTENCY_CACHE_MISSES = Counter(
    "idempotency_cache_misses_total", "Total idempotency cache misses"
)
IDEMPOTENCY_LOCK_ACQUIRED = Counter(
    "idempotency_lock_acquired_total", "Total idempotency locks acquired"
)
IDEMPOTENCY_LOCK_CONFLICTS = Counter(
    "idempotency_lock_conflicts_total", "Total idempotency lock conflicts"
)
IDEMPOTENCY_PROCESSING_TIME = Histogram(
    "idempotency_processing_seconds", "Time spent in idempotency middleware"
)


async def _generate_fingerprint(request: Request) -> str:
    """
    🚀 SINGULARITY: Fast-path fingerprinting.
    Prefers X-Idempotency-Key header to avoid expensive body hashing.
    """
    # 1. Fast-Path: Client-provided idempotency key
    idempotency_key = request.headers.get("X-Idempotency-Key")
    if idempotency_key:
        # Use a sub-hash or just the key if it's already a UUID/Hash
        return f"hdr:{idempotency_key}"

    # 2. Slow-Path: Hash the entire request context
    hasher = hashlib.sha256()
    hasher.update(request.method.encode())
    hasher.update(request.url.path.encode())

    # 🚀 SOTA: Only hash critical headers
    for h in ["authorization", "content-type"]:
        val = request.headers.get(h, "")
        hasher.update(f"{h}:{val}".encode())

    # Read body for mutations
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        hasher.update(body)

    return f"ctx:{hasher.hexdigest()}"


class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        expiry: int = 86400,  # 24 hours
        lock_timeout: int = 60,  # 1 minute
    ):
        super().__init__(app)
        self.redis = redis_client
        self.expiry = expiry
        self.lock_timeout = lock_timeout

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only apply to mutations by default unless configured otherwise
        if request.method not in ("POST", "PUT", "PATCH") and not request.headers.get(
            "X-Idempotency-Key"
        ):
            IDEMPOTENCY_REQUESTS_TOTAL.labels(status="skipped").inc()
            return await call_next(request)

        start_time = time.time()

        # Generate fingerprint (using fast-path if possible)
        fingerprint = await _generate_fingerprint(request)
        cache_key = f"idempotency:res:{fingerprint}"
        lock_key = f"idempotency:lock:{fingerprint}"

        # 1. Check for cached result
        cached_res = await self.redis.get(cache_key)
        if cached_res:
            logger.info("idempotency_cache_hit", path=request.url.path, key=fingerprint)
            IDEMPOTENCY_CACHE_HITS.inc()
            IDEMPOTENCY_REQUESTS_TOTAL.labels(status="cache_hit").inc()
            data = msgspec.json.decode(cached_res)

            IDEMPOTENCY_PROCESSING_TIME.observe(time.time() - start_time)
            return Response(
                content=data["content"],
                status_code=data["status_code"],
                headers={**data["headers"], "X-Idempotency-Cache": "HIT"},
            )

        IDEMPOTENCY_CACHE_MISSES.inc()

        # 2. Try to acquire lock (Distributed Atomic Lock)
        acquired = await self.redis.set(
            lock_key, "LOCKED", ex=self.lock_timeout, nx=True
        )
        if not acquired:
            logger.warning("idempotency_conflict", path=request.url.path)
            IDEMPOTENCY_LOCK_CONFLICTS.inc()
            IDEMPOTENCY_REQUESTS_TOTAL.labels(status="conflict").inc()
            return Response(
                content="Request already in progress. Please wait.", status_code=409
            )

        IDEMPOTENCY_LOCK_ACQUIRED.inc()

        try:
            # 3. Execute request
            response = await call_next(request)

            # 4. Cache successful result
            if response.status_code < 500:
                body_content = b""
                if hasattr(response, "body"):
                    body_content = response.body
                elif hasattr(response, "read"):
                    body_content = await response.read()

                if body_content:
                    cache_data = {
                        "status_code": response.status_code,
                        "content": (
                            body_content.decode("utf-8")
                            if isinstance(body_content, bytes)
                            else body_content
                        ),
                        "headers": dict(response.headers),
                    }

                    # 🚀 SOTA: Atomic multi-set with expiry
                    await self.redis.set(
                        cache_key, msgspec.json.encode(cache_data), ex=self.expiry
                    )
                    IDEMPOTENCY_REQUESTS_TOTAL.labels(status="cached").inc()

                    # Return original response content
                    return Response(
                        content=body_content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )

            IDEMPOTENCY_REQUESTS_TOTAL.labels(status="uncached").inc()
            return response

        finally:
            # 5. Release lock
            await self.redis.delete(lock_key)
            IDEMPOTENCY_PROCESSING_TIME.observe(time.time() - start_time)
