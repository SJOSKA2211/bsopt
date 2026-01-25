import time
from typing import Optional, Tuple
from fastapi import Depends, HTTPException, Request, status
from src.config import settings
from src.utils.cache import get_redis_client
import redis.asyncio as redis # Import redis.asyncio for type hinting

class RateLimiter:
    def __init__(self, requests: int = None, window: int = 60):
        self.requests = requests
        self.window = window

    async def __call__(self, request: Request, redis_client: redis.Redis = Depends(get_redis_client)):
        """
        Rate limiting dependency using Redis for distributed rate limiting.
        """
        if not redis_client:
            # Fallback if Redis is not available, or raise an exception
            # For now, we'll allow all requests but log a warning
            request.state.rate_limit_remaining = 999999 # Unlimited
            request.state.rate_limit_limit = 999999
            request.state.rate_limit_reset = int(time.time() + 60)
            return

        # Prefer user_id if authenticated, otherwise use IP
        user = getattr(request.state, "user", None)

        if self.requests is not None:
             # Explicit limit (e.g. for login/register)
             # Use IP if user is not available
             identifier = str(user.id) if user else (request.client.host if request.client else "unknown")
             limit = self.requests
             window = self.window
             # Include path in key for specific endpoint limits
             key = f"rate_limit:{identifier}:{request.url.path}:{int(time.time()) // window}"
        else:
             # Default tier-based limit (Global limit across endpoints using this dependency)
             identifier = str(user.id) if user else (request.client.host if request.client else "unknown")
             tier = getattr(user, "tier", "free") if user else "free"
             limit = settings.rate_limit_tiers.get(tier, 100)
             window = 60 # Default window for tier-based
             key = f"rate_limit:{identifier}:{int(time.time()) // window}"

        if limit == 0:  # 0 means unlimited
            request.state.rate_limit_remaining = 999999
            request.state.rate_limit_limit = 999999
            request.state.rate_limit_reset = int(time.time() + window)
            return

        # Use a fixed window counter approach
        current_time = int(time.time())

        # Use Redis pipeline for atomic operations
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window) # Ensure key expires after the window

        count, _ = await pipe.execute()

        remaining = max(0, limit - count)
        retry_after = 0
        if count > limit:
            # Get the time left until the current window ends
            ttl = await redis_client.ttl(key)
            if ttl > 0:
                retry_after = ttl
            else:
                # Fallback if TTL somehow isn't set, assume end of current window
                retry_after = window - (current_time % window)

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"X-RateLimit-Limit": str(limit),
                         "X-RateLimit-Remaining": str(remaining),
                         "X-RateLimit-Reset": str(current_time + retry_after), # Approximate reset time
                         "Retry-After": str(retry_after)}
            )

        # Store rate limit info in request state for response headers if desired by middleware
        request.state.rate_limit_limit = limit
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = int(time.time()) + window # Approximate next window start

# Default instance for backward compatibility
rate_limit = RateLimiter()
