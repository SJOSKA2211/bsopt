import hashlib
from collections.abc import Callable
from functools import wraps

import msgspec
import structlog
from fastapi import Request, Response

from src.utils.cache import get_redis

logger = structlog.get_logger()


def generate_key(request: Request, prefix: str) -> str:
    """Generate a consistent cache key based on path and query params."""
    # Create a deterministic key from query parameters
    query_items = sorted(request.query_params.items())
    query_str = "&".join(f"{k}={v}" for k, v in query_items)

    # Construct base key
    base = f"{prefix}:{request.url.path}:{query_str}"

    # Hash for safety and length
    return f"{prefix}:{hashlib.sha256(base.encode()).hexdigest()}"


def cached_endpoint(prefix: str = "api_cache", ttl: int = 60):
    """
    Decorator for FastAPI endpoints to cache responses in Redis.
    Serialized using msgspec for performance.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                request = kwargs.get("request")

            # If we can't find the request, skip caching (or raise)
            if not request:
                logger.warning(
                    "cache_skipped_no_request_object", endpoint=func.__name__
                )
                return await func(*args, **kwargs)

            redis = get_redis()
            if not redis:
                return await func(*args, **kwargs)

            cache_key = generate_key(request, prefix)

            # 1. Try Fetch
            try:
                cached = await redis.get(cache_key)
                if cached:
                    logger.debug("api_cache_hit", key=cache_key)
                    return Response(content=cached, media_type="application/json")
            except Exception as e:
                logger.error("api_cache_read_error", error=str(e))

            # 2. Compute
            response = await func(*args, **kwargs)

            # 3. Cache (only if it's a valid response)
            # Support both direct Pydantic models (msgspec friendly) and Response objects?
            # For simplicity, we assume the endpoint returns data that FastAPI will serialize.
            # BUT, standard FastAPI decorators serialize AFTER the function returns.
            # To cache the *result*, we might need to handle serialization ourselves or cache the Pydantic model dump.

            # Implementation Choice: Cache the serialized JSON if possible.
            # If the function returns a Pydantic model or dict, we serialize it.
            try:
                if hasattr(response, "json"):  # Pydantic v2
                    data = response.model_dump_json()
                elif isinstance(response, dict) or isinstance(response, list):
                    data = msgspec.json.encode(response)
                else:
                    # Fallback or skip
                    return response

                await redis.setex(cache_key, ttl, data)
            except Exception as e:
                logger.error("api_cache_write_error", error=str(e))

            return response

        return wrapper

    return decorator
