"""
Redis Caching Strategy
======================

Implements a multi-layer caching strategy using Redis to improve API performance.
Optimized for 1000+ concurrent users with connection pooling and keepalive.
"""

import asyncio
import hashlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from typing import Any

import msgspec
import orjson
import structlog
from cachetools import TTLCache
from redis.asyncio import Redis, RedisError

from src.pricing.models import BSParameters, OptionGreeks

logger = structlog.get_logger(__name__)

_redis: Redis | None = None


def get_redis() -> Redis | None:
    """Get or initialize the global Redis client instance."""
    global _redis
    if _redis is None:
        from src.config import settings

        try:
            _redis = Redis.from_url(settings.REDIS_URL, decode_responses=False)
            logger.info("redis_client_initialized", url=settings.REDIS_URL)
        except Exception as e:
            logger.error("redis_initialization_failed", error=str(e))
            return None
    return _redis


def generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate a deterministic cache key using ultra-fast msgspec serialization.
    """
    # msgspec is the fastest serialization library available for Python
    param_json = msgspec.json.encode(kwargs)
    return f"{prefix}:{hashlib.sha256(param_json).hexdigest()}"


class PricingCache:
    async def get_option_price(
        self, params: BSParameters, option_type: str, method: str
    ) -> float | None:
        redis = get_redis()
        if redis is None:
            return None
        key = generate_cache_key(f"{method}:{option_type}", **asdict(params))
        try:
            val = await redis.get(key)
            if val:
                return float(msgspec.json.decode(val))
            return None
        except (AttributeError, RedisError, ValueError) as e:
            logger.error("cache_get_price_failed", error=str(e), key=key)
            return None

    async def set_option_price(
        self,
        params: BSParameters,
        option_type: str,
        method: str,
        price: float,
        ttl: int = 3600,
    ):
        redis = get_redis()
        if redis is None:
            return False
        key = generate_cache_key(f"{method}:{option_type}", **asdict(params))
        try:
            await redis.setex(key, ttl, msgspec.json.encode(float(price)))
            return True
        except (AttributeError, RedisError, TypeError) as e:
            logger.error("cache_set_price_failed", error=str(e), key=key)
            return False

    async def get_greeks(
        self, params: BSParameters, option_type: str
    ) -> OptionGreeks | None:
        """Retrieve cached Greeks."""
        redis = get_redis()
        if redis is None:
            return None
        key = generate_cache_key(f"greeks:{option_type}", **asdict(params))
        try:
            val = await redis.get(key)
            if val:
                data = msgspec.json.decode(val)
                return OptionGreeks(**data)
            return None
        except Exception as e:
            logger.error("cache_get_greeks_failed", error=str(e), key=key)
            return None

    async def set_greeks(
        self,
        params: BSParameters,
        option_type: str,
        greeks: OptionGreeks,
        ttl: int = 3600,
    ):
        """Cache Greeks."""
        redis = get_redis()
        if redis is None:
            return False
        key = generate_cache_key(f"greeks:{option_type}", **asdict(params))
        try:
            await redis.setex(key, ttl, msgspec.json.encode(asdict(greeks)))
            return True
        except Exception as e:
            logger.error("cache_set_greeks_failed", error=str(e), key=key)
            return False


def multi_layer_cache(
    prefix: str, maxsize: int = 1000, ttl: int = 60, validation_model: Any = None
):
    """
    Decorator for multi-layer caching with SOTA X-Fetch (Probabilistic Early Recomputation).
    Layer 1: Local In-Memory LRU
    Layer 2: Distributed Redis
    """
    l1_cache = TTLCache(maxsize=maxsize, ttl=ttl)
    beta = 1.0  # X-Fetch coefficient (higher means more aggressive early refresh)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import math
            import random

            key_params = kwargs.copy()
            for i, arg in enumerate(args[1:]):
                key_params[f"arg_{i}"] = arg

            cache_key = generate_cache_key(prefix, **key_params)

            # 1. L1 Check (with X-Fetch logic for local memory?)
            # Usually X-Fetch is most beneficial for network-bound L2
            if cache_key in l1_cache:
                return l1_cache[cache_key]

            # 2. L2 Check (Redis) with X-Fetch implementation
            redis = get_redis()
            recompute = False
            cached_val = None

            if redis:
                try:
                    # SOTA: Fetch value AND remaining TTL
                    pipe = redis.pipeline()
                    pipe.get(cache_key)
                    pipe.pttl(cache_key)
                    cached_val, remaining_ms = await pipe.execute()

                    if cached_val:
                        # X-Fetch formula: now - (delta * beta * log(random())) > expiry
                        # delta is the time it took to compute (approximate here)
                        # We use a constant 'beta' and random jitter to trigger refresh
                        delta_ms = 100  # Assume 100ms computation time average
                        if (
                            remaining_ms > 0
                            and (
                                remaining_ms
                                - delta_ms * beta * math.log(random.random())
                            )
                            < 0
                        ):
                            logger.info(
                                "x_fetch_triggered_early_refresh", key=cache_key
                            )
                            recompute = True
                        else:
                            val = msgspec.json.decode(cached_val)
                            if validation_model and isinstance(val, dict):
                                val = validation_model(**val)
                            l1_cache[cache_key] = val
                            return val
                except Exception as e:
                    logger.warning("l2_cache_read_failed", error=str(e))

            # 3. Execute (if not in cache OR x-fetch triggered)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                from anyio.to_thread import run_sync

                result = await run_sync(func, *args, **kwargs)

            # 4. Update Caches
            l1_cache[cache_key] = result
            if redis:
                try:
                    # SOTA: msgspec is extremely fast for complex objects
                    await redis.setex(cache_key, 3600, msgspec.json.encode(result))
                except Exception as e:
                    logger.warning("l2_cache_write_failed", error=str(e))

            return result

        return wrapper

    return decorator


class RateLimitTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    pricing_requests_per_minute: int
    burst_size: int


RATE_LIMIT_CONFIGS = {
    RateLimitTier.FREE: RateLimitConfig(100, 50, 10),
    RateLimitTier.PRO: RateLimitConfig(1000, 500, 50),
    RateLimitTier.ENTERPRISE: RateLimitConfig(10000, 5000, 500),
}

pricing_cache = PricingCache()


class RateLimiter:
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        tier: RateLimitTier | str = RateLimitTier.FREE,
    ) -> bool:
        redis = get_redis()
        if redis is None:
            return True

        # Convert string to Enum if needed
        if isinstance(tier, str):
            try:
                tier = RateLimitTier(tier.lower())
            except ValueError:
                tier = RateLimitTier.FREE

        config = RATE_LIMIT_CONFIGS[tier]
        limit = (
            config.pricing_requests_per_minute
            if "price" in endpoint.lower()
            else config.requests_per_minute
        )
        key = f"rl:{user_id}:{endpoint}"
        now = int(time.time())
        window = now // 60
        full_key = f"{key}:{window}"
        try:
            pipe = redis.pipeline()
            pipe.incr(full_key)
            pipe.expire(full_key, 120)
            results = await pipe.execute()
            return bool(results[0] <= (limit + config.burst_size))
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e), user_id=user_id)
            return True


rate_limiter = RateLimiter()


async def warm_cache():
    """Pre-warm cache with common option parameters in parallel."""
    from src.pricing.black_scholes import BlackScholesEngine

    # Common scenarios
    spots = [100.0]
    strikes = [90.0, 100.0, 110.0]
    maturities = [0.1, 0.5, 1.0]
    vols = [0.2, 0.4]

    logger.info("warming_cache_start")
    tasks = []
    for s in spots:
        for k in strikes:
            for t in maturities:
                for v in vols:
                    params = BSParameters(s, k, t, v, 0.05, 0.02)
                    price = BlackScholesEngine.price_options(
                        spot=s,
                        strike=k,
                        maturity=t,
                        volatility=v,
                        rate=0.05,
                        dividend=0.02,
                        option_type="call",
                    )
                    tasks.append(
                        pricing_cache.set_option_price(
                            params, "call", "bs_unified", float(price)
                        )
                    )

    if tasks:
        await asyncio.gather(*tasks)
    logger.info("warming_cache_complete", count=len(tasks))


class IdempotencyManager:
    PREFIX = "idem:"

    async def check_and_set(self, key: str, ttl: int = 3600) -> bool:
        """
        Check if key exists. If not, set it and return True.
        If it exists, return False.
        """
        redis = get_redis()
        if redis is None:
            return True  # Fail open if redis is down

        full_key = f"{self.PREFIX}{key}"
        # set with nx=True only sets if it doesn't exist
        result = await redis.set(full_key, "1", ex=ttl, nx=True)
        return bool(result)


idempotency_manager = IdempotencyManager()


class DatabaseQueryCache:
    PREFIX = "db:"

    async def get_user(self, user_id: str) -> dict | None:
        redis = get_redis()
        if redis is None:
            return None
        try:
            val = await redis.get(f"{self.PREFIX}user:{user_id}")
            return orjson.loads(val) if val else None
        except Exception as e:
            logger.error("db_cache_get_user_failed", error=str(e), user_id=user_id)
            return None

    async def set_user(self, user_id: str, user_data: dict, ttl: int = 300):
        redis = get_redis()
        if redis is None:
            return False
        try:
            await redis.setex(
                f"{self.PREFIX}user:{user_id}", ttl, orjson.dumps(user_data)
            )
            return True
        except Exception as e:
            logger.error("db_cache_set_user_failed", error=str(e), user_id=user_id)
            return False


db_cache = DatabaseQueryCache()


# --- Real-time updates support ---
redis_channel_updates: str = "pricing_updates"


async def publish_to_redis(channel: str, message: dict[str, Any]):
    """Publish a message to a Redis channel using orjson."""
    redis = get_redis()
    if redis is not None:
        try:
            encoded_message = orjson.dumps(message)
            await redis.publish(channel, encoded_message)
            logger.debug("redis_publish_success", channel=channel)
        except Exception as e:
            logger.error("redis_publish_failed", error=str(e), channel=channel)


async def get_redis_client() -> Redis:
    """FastAPI dependency to get the Redis client."""
    redis = get_redis()
    if redis is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Redis client not initialized")
        return redis


async def init_redis_cache(**kwargs):
    """Initialize the Redis cache during startup."""
    redis = get_redis()
    if redis:
        try:
            await redis.ping()
            logger.info("redis_cache_initialized")
        except Exception as e:
            logger.error("redis_cache_init_failed", error=str(e))


async def close_redis_cache():
    """Close the Redis client connection."""
    global _redis
    if _redis:
        try:
            await _redis.aclose()
            _redis = None
            logger.info("redis_cache_closed")
        except Exception as e:
            logger.error("redis_cache_close_failed", error=str(e))
