"""
Distributed Rate Limiting with Redis Token Bucket.
Atomic implementation using LUA scripts for high-performance decrementing.
"""

import time
from enum import StrEnum

import structlog
from redis.asyncio import Redis

from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class RateLimitTier(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Default configurations: (capacity, refill_rate_per_sec)
TIER_CONFIGS = {
    RateLimitTier.FREE: (100, 1),           # 100 requests, refill 1/sec (60/min)
    RateLimitTier.PRO: (1000, 10),         # 1000 requests, refill 10/sec (600/min)
    RateLimitTier.ENTERPRISE: (10000, 100), # 10000 requests, refill 100/sec (6000/min)
}

LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4] or 1)

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if not tokens then
    tokens = capacity
    last_refill = now
else
    local elapsed = now - last_refill
    local refill = elapsed * refill_rate
    tokens = math.min(capacity, tokens + refill)
    last_refill = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, 60)
    return 1
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, 60)
    return 0
end
"""

class RedisTokenBucketLimiter:
    def __init__(self):
        self._lua_script = None

    async def _get_script(self, redis: Redis):
        if self._lua_script is None:
            self._lua_script = redis.register_script(LUA_TOKEN_BUCKET)
        return self._lua_script

    async def is_allowed(self, user_id: str, endpoint: str, tier: RateLimitTier = RateLimitTier.FREE) -> bool:
        redis = get_redis()
        if redis is None:
            return True # Fail open

        capacity, refill_rate = TIER_CONFIGS.get(tier, TIER_CONFIGS[RateLimitTier.FREE])
        key = f"rl:tb:{user_id}:{endpoint}"
        now = time.time()
        
        script = await self._get_script(redis)
        try:
            result = await script(keys=[key], args=[capacity, refill_rate, now, 1])
            return bool(result)
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e), user_id=user_id)
            return True # Fail open

limiter = RedisTokenBucketLimiter()
