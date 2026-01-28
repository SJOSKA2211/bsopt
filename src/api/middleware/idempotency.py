import hashlib
import orjson
import asyncio
from typing import Any, Optional, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, StreamingResponse
import redis.asyncio as redis
import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger()

# ... (metrics remain same)

async def _generate_fingerprint(request: Request) -> str:
    """
    Generates a unique SHA-256 fingerprint for a request.
    Optimized for performance.
    """
    hasher = hashlib.sha256()
    hasher.update(request.method.encode())
    hasher.update(request.url.path.encode())
    
    # Sort and hash headers efficiently
    header_items = sorted(request.headers.items())
    for k, v in header_items:
        hasher.update(f"{k.lower()}:{v}".encode())
        
    # Read body once
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
                data = orjson.loads(cached_res)
                return Response(
                    content=data["content"],
                    status_code=data["status_code"],
                    headers={**data["headers"], "X-Idempotency-Cache": "HIT"}
                )
            
            IDEMPOTENCY_CACHE_MISSES.inc()

            # 2. Try to acquire lock (Distributed Lock)
            acquired = await self.redis.set(lock_key, "LOCKED", ex=self.lock_timeout, nx=True)
            if not acquired:
                logger.warning("idempotency_conflict", path=request.url.path)
                IDEMPOTENCY_LOCK_CONFLICTS.inc()
                IDEMPOTENCY_REQUESTS_TOTAL.labels(status='conflict').inc()
                return Response(
                    content="Request already in progress. Please wait.",
                    status_code=409
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
                        
                        # Use orjson for high-speed serialization
                        await self.redis.set(cache_key, orjson.dumps(cache_data), ex=self.expiry)
                        IDEMPOTENCY_REQUESTS_TOTAL.labels(status='cached').inc()
                        return full_response
                
                IDEMPOTENCY_REQUESTS_TOTAL.labels(status='uncached').inc()
                return response

            finally:
                # 5. Release lock
                await self.redis.delete(lock_key)