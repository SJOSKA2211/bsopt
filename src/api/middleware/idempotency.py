import hashlib
import json
import asyncio
from typing import Any, Optional, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse
import redis.asyncio as redis
import structlog
from prometheus_client import Counter, Gauge, Histogram # Import Prometheus client metrics

logger = structlog.get_logger()

# Prometheus Metrics for Idempotency
IDEMPOTENCY_REQUESTS_TOTAL = Counter('idempotency_requests_total', 'Total requests processed by idempotency middleware', ['status'])
IDEMPOTENCY_CACHE_HITS = Counter('idempotency_cache_hits_total', 'Total cache hits in idempotency middleware')
IDEMPOTENCY_CACHE_MISSES = Counter('idempotency_cache_misses_total', 'Total cache misses in idempotency middleware')
IDEMPOTENCY_LOCK_ACQUIRED = Counter('idempotency_lock_acquired_total', 'Total times idempotency lock was acquired')
IDEMPOTENCY_LOCK_CONFLICTS = Counter('idempotency_lock_conflicts_total', 'Total times idempotency lock could not be acquired (conflict)')
IDEMPOTENCY_PROCESSING_TIME = Histogram('idempotency_processing_seconds', 'Request processing time in idempotency middleware')

async def _generate_fingerprint(request: Request) -> str:
    """
    Generates a unique SHA-256 fingerprint for a request.
    """
    hasher = hashlib.sha256()
    hasher.update(request.method.encode())
    hasher.update(request.url.path.encode())
    
    header_items = []
    for k, v in request.headers.items():
        header_items.append(f"{k.lower()}:{v}")
    
    for item in sorted(header_items):
        hasher.update(item.encode())
        
    body = await request.body()
    hasher.update(body)
    
    return hasher.hexdigest()

class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app, 
        redis_client: redis.Redis,
        expiry: int = 86400, # 24 hours
        lock_timeout: int = 60 # 1 minute
    ):
        super().__init__(app)
        self.redis = redis_client
        self.expiry = expiry
        self.lock_timeout = lock_timeout

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only apply to mutations
        if request.method not in ("POST", "PUT", "PATCH"):
            IDEMPOTENCY_REQUESTS_TOTAL.labels(status='skipped').inc()
            return await call_next(request)

        # Generate fingerprint
        fingerprint = await _generate_fingerprint(request)
        cache_key = f"idempotency:res:{fingerprint}"
        lock_key = f"idempotency:lock:{fingerprint}"

        with IDEMPOTENCY_PROCESSING_TIME.time():
            # 1. Check for cached result
            cached_res = await self.redis.get(cache_key)
            if cached_res:
                logger.info("idempotency_cache_hit", path=request.url.path)
                IDEMPOTENCY_CACHE_HITS.inc()
                IDEMPOTENCY_REQUESTS_TOTAL.labels(status='cache_hit').inc()
                data = json.loads(cached_res)
                return Response(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers={**data["headers"], "X-Idempotency-Cache": "HIT"}
                )
            
            IDEMPOTENCY_CACHE_MISSES.inc()

            # 2. Try to acquire lock (Distributed Lock)
            # setnx returns True if key was set (acquired), False otherwise
            acquired = await self.redis.setnx(lock_key, "LOCKED")
            if not acquired:
                logger.warning("idempotency_conflict", path=request.url.path)
                IDEMPOTENCY_LOCK_CONFLICTS.inc()
                IDEMPOTENCY_REQUESTS_TOTAL.labels(status='conflict').inc()
                return Response(
                    content="Request already in progress. Please wait.",
                    status_code=409
                )
            
            IDEMPOTENCY_LOCK_ACQUIRED.inc()
            # Ensure lock expires even if we crash
            await self.redis.expire(lock_key, self.lock_timeout)

            try:
                # 3. Execute request
                response = await call_next(request)

                # 4. Cache successful result
                # We don't cache 5xx or streaming responses for simplicity
                is_streaming = isinstance(response, StreamingResponse) or hasattr(response, "body_iterator")
                if response.status_code < 500 and not is_streaming:
                    # Consume body to cache it
                    body_content = await response.read()
                    
                    # Re-create response because body_iterator was consumed
                    full_response = Response(
                        content=body_content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )

                    cache_data = {
                        "status_code": full_response.status_code,
                        "content": body_content.decode('utf-8') if isinstance(body_content, bytes) else body_content,
                        "headers": dict(full_response.headers)
                    }
                    
                    await self.redis.set(cache_key, json.dumps(cache_data), ex=self.expiry)
                    IDEMPOTENCY_REQUESTS_TOTAL.labels(status='cached').inc()
                    return full_response
                
                IDEMPOTENCY_REQUESTS_TOTAL.labels(status='uncached').inc()
                return response

            finally:
                # 5. Release lock
                await self.redis.delete(lock_key)