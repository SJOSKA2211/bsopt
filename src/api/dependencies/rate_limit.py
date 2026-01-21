"""
Rate Limiting Dependency
========================

Provides dependencies for rate limiting sensitive endpoints like login.
"""

import structlog
from fastapi import HTTPException, Request, status
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class LoginRateLimiter:
    """
    Rate limiter for login endpoints based on IP address.

    Usage:
        @router.post("/login", dependencies=[Depends(LoginRateLimiter(max_attempts=5, window_seconds=300))])
    """

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds

    async def __call__(self, request: Request):
        redis = get_redis()
        if not redis:
            # Open fail if Redis is not available
            logger.warning("Redis not available, skipping rate limit check")
            return

        # Use request.client.host to avoid IP spoofing via X-Forwarded-For
        # We assume the infrastructure (Uvicorn/Gunicorn) is configured with ProxyHeadersMiddleware
        # if behind a trusted proxy.
        ip = request.client.host if request.client else "unknown"
        key = f"login_limit:{ip}"

        try:
            # Atomic INCR + EXPIRE using Lua script
            # If the key is new (attempts == 1), set the expiration.
            # Otherwise, just return the current value.
            lua_script = """
            local current = redis.call("INCR", KEYS[1])
            if current == 1 then
                redis.call("EXPIRE", KEYS[1], ARGV[1])
            end
            return current
            """

            attempts = await redis.eval(lua_script, 1, key, self.window_seconds)

            if attempts > self.max_attempts:
                logger.warning("login_rate_limit_exceeded", ip=ip, attempts=attempts)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("rate_limit_error", error=str(e))
            # Fail open
