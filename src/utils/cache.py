"""
Redis Caching Strategy
======================

Implements a multi-layer caching strategy using Redis to improve API performance.
Optimized for 1000+ concurrent users with connection pooling and keepalive.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, TypeVar

import numpy as np

try:
    import redis.asyncio as aioredis
    from redis.asyncio import ConnectionPool, Redis
    if hasattr(aioredis, "RedisError") and isinstance(aioredis.RedisError, type):
        RedisError = aioredis.RedisError
    else:
        class RedisError(Exception): pass
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore
    Redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    class RedisError(Exception): pass

from src.pricing.black_scholes import BSParameters, OptionGreeks

logger = logging.getLogger(__name__)

T = TypeVar("T")
_redis_pool: Optional[Redis] = None


class RateLimitTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimitConfig:
    requests_per_minute: int
    requests_per_hour: int
    burst_size: int
    pricing_requests_per_minute: int


RATE_LIMIT_CONFIGS: Dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.FREE: RateLimitConfig(60, 1000, 10, 30),
    RateLimitTier.PRO: RateLimitConfig(300, 10000, 50, 150),
    RateLimitTier.ENTERPRISE: RateLimitConfig(1000, 50000, 200, 500),
}


async def init_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    max_connections: int = 100,
) -> Optional[Redis]:
    global _redis_pool
    if aioredis is None:
        return None
    try:
        # Optimized connection pool for high throughput
        pool = ConnectionPool.from_url(
            f"redis://{host}:{port}/{db}",
            password=password,
            max_connections=max_connections,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            socket_keepalive=True,
            health_check_interval=30.0,
        )
        _redis_pool = Redis(connection_pool=pool)
        await _redis_pool.ping()
        logger.info(
            f"Redis cache initialized with pool (max={max_connections}): {host}:{port}/{db}"
        )
        return _redis_pool
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        _redis_pool = None
        return None


async def close_redis_cache() -> None:
    global _redis_pool
    if _redis_pool:
        try:
            await _redis_pool.close()
        finally:
            _redis_pool = None


def get_redis() -> Optional[Redis]:
    return _redis_pool


def generate_cache_key(prefix: str, **kwargs) -> str:
    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    sorted_params = {k: kwargs[k] for k in sorted(kwargs.keys())}
    param_json = json.dumps(sorted_params, sort_keys=True, default=json_serializer)
    return f"{prefix}:{hashlib.sha256(param_json.encode()).hexdigest()}"


class PricingCache:
    async def get_option_price(
        self, params: BSParameters, option_type: str, method: str
    ) -> Optional[float]:
        redis = get_redis()
        if redis is None:
            return None
        key = generate_cache_key(f"{method}:{option_type}", **asdict(params))
        try:
            val = await redis.get(key)
            if val:
                return float(json.loads(val))
            return None
        except (AttributeError, RedisError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get option price from cache: {e}")
            return None

    async def set_option_price(
        self, params: BSParameters, option_type: str, method: str, price: float, ttl: int = 3600
    ):
        redis = get_redis()
        if redis is None:
            return False
        key = generate_cache_key(f"{method}:{option_type}", **asdict(params))
        try:
            await redis.setex(key, ttl, json.dumps(float(price)))
            return True
        except (AttributeError, RedisError, TypeError) as e:
            logger.error(f"Failed to set option price in cache: {e}")
            return False

    async def get_greeks(self, params: BSParameters, option_type: str) -> Optional[OptionGreeks]:
        """Retrieve cached Greeks."""
        redis = get_redis()
        if redis is None:
            return None
        key = generate_cache_key(f"greeks:{option_type}", **asdict(params))
        try:
            val = await redis.get(key)
            if val:
                data = json.loads(val)
                return OptionGreeks(**data)
            return None
        except Exception as e:
            logger.error(f"Failed to get greeks from cache: {e}")
            return None

    async def set_greeks(
        self, params: BSParameters, option_type: str, greeks: OptionGreeks, ttl: int = 3600
    ):
        """Cache Greeks."""
        redis = get_redis()
        if redis is None:
            return False
        key = generate_cache_key(f"greeks:{option_type}", **asdict(params))
        try:
            await redis.setex(key, ttl, json.dumps(asdict(greeks)))
            return True
        except Exception as e:
            logger.error(f"Failed to set greeks in cache: {e}")
            return False


pricing_cache = PricingCache()


class RateLimiter:
    async def check_rate_limit(
        self, user_id: str, endpoint: str, tier: RateLimitTier = RateLimitTier.FREE
    ) -> bool:
        redis = get_redis()
        if redis is None:
            return True
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
            logger.error(f"Rate limiter failed: {e}")
            return True


rate_limiter = RateLimiter()


async def warm_cache():
    """Pre-warm cache with common option parameters."""
    from src.pricing.black_scholes import BlackScholesEngine

    # Common scenarios
    spots = [100.0]
    strikes = [90.0, 100.0, 110.0]
    maturities = [0.1, 0.5, 1.0]
    vols = [0.2, 0.4]

    logger.info("Warming pricing cache...")
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
                    # price is a float here for single value
                    await pricing_cache.set_option_price(params, "call", "bs_unified", float(price))
    logger.info("Cache warming complete.")


class IdempotencyManager:
    PREFIX = "idem:"

    async def check_and_set(self, key: str, ttl: int = 3600) -> bool:
        """
        Check if key exists. If not, set it and return True.
        If it exists, return False.
        """
        redis = get_redis()
        if redis is None:
            return True # Fail open if redis is down
        
        full_key = f"{self.PREFIX}{key}"
        # set with nx=True only sets if it doesn't exist
        result = await redis.set(full_key, "1", ex=ttl, nx=True)
        return bool(result)

idempotency_manager = IdempotencyManager()


class DatabaseQueryCache:
    PREFIX = "db:"

    async def get_user(self, user_id: str) -> Optional[Dict]:
        redis = get_redis()
        if redis is None:
            return None
        try:
            val = await redis.get(f"{self.PREFIX}user:{user_id}")
            return json.loads(val) if val else None
        except Exception as e:
            logger.error(f"Failed to get user from database cache: {e}")
            return None

    async def set_user(self, user_id: str, user_data: Dict, ttl: int = 300):
        redis = get_redis()
        if redis is None:
            return False
        try:
            await redis.setex(
                f"{self.PREFIX}user:{user_id}", ttl, json.dumps(user_data, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set user in database cache: {e}")
            return False


db_cache = DatabaseQueryCache()


# --- Real-time updates support ---
redis_channel_updates: str = "pricing_updates"


async def publish_to_redis(channel: str, message: Dict[str, Any]):
    """Publish a message to a Redis channel."""
    redis = get_redis()
    if redis is not None:
        try:
            encoded_message = json.dumps(message)
            await redis.publish(channel, encoded_message)
            logger.debug(f"Published to Redis channel '{channel}'")
        except Exception as e:
            logger.error(f"Failed to publish to Redis channel '{channel}': {e}")


async def get_redis_client() -> Redis:
    """FastAPI dependency to get the Redis client."""
    redis = get_redis()
    if redis is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Redis client not initialized")
    return redis
