import time

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request, status

from src.shared.config import settings
from src.shared.lua_scripts import TOKEN_BUCKET_RL
from src.shared.utils.cache import get_redis_client

logger = structlog.get_logger(__name__)


async def rate_limit(request: Request, redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Advanced Token Bucket Rate Limiting using Redis LUA.
    Provides scalable protection for inbound APIs.
    """
    if not redis_client:
        logger.error("rate_limit_redis_client_none")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable.",
        )

    # Identifier logic (User ID or IP)
    user = getattr(request.state, "user", None)
    identifier = str(user.id) if user else (request.client.host if request.client else "unknown")
    tier = getattr(user, "tier", "free") if user else "free"

    # Define capacity and fill rate based on tier
    limit_settings = settings.rate_limit_tiers.get(tier, {"capacity": 100, "fill_rate": 1})
    if isinstance(limit_settings, int):
        capacity = limit_settings
        fill_rate = max(1, capacity // 60)
    else:
        capacity = limit_settings.get("capacity", 100)
        fill_rate = limit_settings.get("fill_rate", 1)

    if capacity == 0:
        return

    key = f"rate_limit:token_bucket:{identifier}"
    now_ms = int(time.time() * 1000)

    # Execute LUA script for atomic token bucket check
    # Script returns {allowed (0/1), current_tokens}
    try:
        allowed, current_tokens = await redis_client.eval(
            TOKEN_BUCKET_RL, 1, key, capacity, fill_rate, now_ms, 1
        )
    except Exception as e:
        logger.error("rate_limit_lua_failed", error=str(e))
        return  # Fail open on script error to not block users, but log it

    if not allowed:
        logger.warning("rate_limit_exceeded", identifier=identifier, tokens=current_tokens)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
            headers={
                "X-RateLimit-Limit": str(capacity),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + max(1, capacity // fill_rate)),
                "Retry-After": str(max(1, 1 // fill_rate)),
            },
        )

    # Store metadata for headers
    request.state.rate_limit_limit = capacity
    request.state.rate_limit_remaining = current_tokens
    request.state.rate_limit_reset = int(time.time()) + max(1, capacity // fill_rate)