import time
import uuid

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request, status

from src.config import settings
from src.shared.lua_scripts import SLIDING_WINDOW_RL
from src.utils.cache import get_redis_client

logger = structlog.get_logger(__name__)


async def rate_limit(request: Request, redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Advanced Sliding Window Rate Limiting using Redis LUA.
    Provides sub-second precision and atomicity.
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

    limit = settings.rate_limit_tiers.get(tier, 100)
    window_ms = 60 * 1000  # 60s window in milliseconds

    if limit == 0:
        return

    key = f"rate_limit:sliding:{identifier}"
    now_ms = int(time.time() * 1000)
    request_id = str(uuid.uuid4())

    # Execute LUA script for atomic sliding window check
    # Script returns {allowed (0/1), current_count}
    try:
        allowed, current_count = await redis_client.eval(
            SLIDING_WINDOW_RL, 1, key, window_ms, limit, now_ms, request_id
        )
    except Exception as e:
        logger.error("rate_limit_lua_failed", error=str(e))
        return  # Fail open on script error to not block users, but log it

    remaining = max(0, limit - current_count)

    if not allowed:
        logger.warning("rate_limit_exceeded", identifier=identifier, count=current_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again in a minute.",
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + 60),
                "Retry-After": "60",
            },
        )

    # Store metadata for headers
    request.state.rate_limit_limit = limit
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_reset = int(time.time()) + 60
