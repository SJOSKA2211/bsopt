import hashlib
from collections.abc import Callable
from functools import wraps

import msgspec
import structlog
from fastapi import Request, Response

from src.utils.cache import get_redis

logger = structlog.get_logger()


def generate_key(request: Request, prefix: str) -> str:
    """Generate a high-performance consistent cache key."""
    # OPTIMIZED: Deterministic sorting of query params for stable hashing
    params = sorted(request.query_params.items())

    # Use blake2b for faster hashing than sha256
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(request.url.path.encode())
    for k, v in params:
        hasher.update(f"{k}:{v}".encode())

    return f"{prefix}:{hasher.hexdigest()}"

def cached_endpoint(prefix: str = "api_cache", ttl: int = 60):
    """
    Decorator for FastAPI endpoints with Response-aware caching.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # ... (Request extraction stays same)
            request = next((arg for arg in args if isinstance(arg, Request)), None) or kwargs.get("request")

            if not request:
                return await func(*args, **kwargs)

            redis = get_redis()
            if not redis:
                return await func(*args, **kwargs)

            cache_key = generate_key(request, prefix)

            try:
                cached = await redis.get(cache_key)
                if cached:
                    return Response(content=cached, media_type="application/json", headers={"X-Cache": "HIT"})
            except Exception:
                pass

            response = await func(*args, **kwargs)

            # 3. Intelligent Caching
            try:
                data = None
                if isinstance(response, Response):
                    # OPTIMIZED: Capture body from Response objects
                    data = response.body
                elif hasattr(response, "model_dump_json"):
                    data = response.model_dump_json().encode()
                elif isinstance(response, (dict, list)):
                    data = msgspec.json.encode(response)

                if data:
                    await redis.setex(cache_key, ttl, data)
            except Exception as e:
                logger.error("api_cache_write_error", error=str(e))

            return response
        return wrapper
    return decorator
