import hashlib

import msgspec
import structlog
from fastapi import Request, Response
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

logger = structlog.get_logger()


async def _generate_fingerprint(request: Request) -> str:
    """Calculate a request fingerprint safely."""
    # Use existing fingerprint in state if available (from earlier middleware)
    if hasattr(request.state, "idempotency_key"):
        return request.state.idempotency_key

    idempotency_key = request.headers.get("X-Idempotency-Key")
    if idempotency_key:
        return f"hdr:{idempotency_key}"

    hasher = hashlib.sha256()
    hasher.update(request.method.encode())
    hasher.update(request.url.path.encode())

    # Only hash body for write operations
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()
        # Starlette hack to allow multiple reads
        request._body = body
        hasher.update(body)

    return f"ctx:{hasher.hexdigest()}"


class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client, expiry: int = 3600, lock_timeout: int = 60):
        super().__init__(app)
        self.redis = redis_client
        self.expiry = expiry
        self.lock_timeout = lock_timeout

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method not in ("POST", "PUT", "PATCH") and not request.headers.get(
            "X-Idempotency-Key"
        ):
            return await call_next(request)

        fingerprint = await _generate_fingerprint(request)
        cache_key = f"idempotency:res:{fingerprint}"
        lock_key = f"idempotency:lock:{fingerprint}"

        # 1. Check Cache
        cached = await self.redis.get(cache_key)
        if cached:
            logger.info("idempotency_cache_hit", key=fingerprint)
            data = msgspec.json.decode(cached)
            headers = data["headers"]
            headers["X-Idempotency-Cache"] = "HIT"
            return Response(
                content=data["content"],
                status_code=data["status_code"],
                headers=headers,
            )

        # 2. Acquire Lock (Simple SETNX)
        if not await self.redis.set(lock_key, "1", nx=True, ex=self.lock_timeout):
            logger.warning("idempotency_lock_conflict", key=fingerprint)
            return Response(content='{"error": "Request already in progress"}', status_code=409)

        try:
            response = await call_next(request)

            # 3. Cache Result (Streaming compatible)
            if response.status_code < 500 and response.status_code != 204:
                if isinstance(response, StreamingResponse):
                    response_body = [chunk async for chunk in response.body_iterator]
                    response.body_iterator = iterate_in_threadpool(iter(response_body))
                    full_body = b"".join(response_body)
                else:
                    full_body = response.body

                if full_body:
                    cache_data = {
                        "status_code": response.status_code,
                        "content": full_body.decode("utf-8", errors="ignore"),
                        "headers": dict(response.headers),
                    }
                    await self.redis.set(cache_key, msgspec.json.encode(cache_data), ex=self.expiry)

                return Response(
                    content=full_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            return response
        finally:
            await self.redis.delete(lock_key)
            await self.redis.delete(lock_key)
