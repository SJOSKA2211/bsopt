"""
Redis Caching Strategy

Implements a multi-layer caching strategy using Redis to improve API performance.
Optimized for 1000+ concurrent users with connection pooling and keepalive.
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import msgspec
import structlog
from cachetools import TTLCache
from redis.asyncio import Redis, RedisError

if TYPE_CHECKING:
    from src.math_kernel.models import BSParameters, OptionGreeks

logger = structlog.get_logger(__name__)

_redis: Redis | None = None


def get_redis() -> Redis | None:
    """Get or initialize the global Redis client instance."""
    global _redis
    if _redis is None:
        from src.shared.config import settings

        try:
            _redis = Redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                max_connections=100,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True,
            )

            logger.info("redis_client_initialized", url=settings.REDIS_URL, max_connections=100)
        except Exception as e:
            logger.error("redis_initialization_failed", error=str(e))
            return None
    return _redis


def generate_cache_key(prefix: str, **kwargs: float | int | str | bool | None) -> str:
    """
    Generate a deterministic cache key using ultra-fast msgspec serialization.
    """
    # msgspec is the fastest serialization library available for Python
    param_json = msgspec.json.encode(kwargs)
    return f"{prefix}:{hashlib.sha256(param_json).hexdigest()}"


class PricingCache:
    async def get_option_price(
        self, params: "BSParameters", option_type: str, method: str
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
        params: "BSParameters",
        option_type: str,
        method: str,
        price: float,
        ttl: int = 3600,
    ) -> bool:
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

    async def get_greeks(self, params: "BSParameters", option_type: str) -> "OptionGreeks | None":
        """Retrieve cached Greeks."""
        from src.math_kernel.models import OptionGreeks

        redis = get_redis()
        if redis is None:
            return None
        key = generate_cache_key(f"greeks:{option_type}", **asdict(params))
        try:
            val = await redis.get(key)
            if val:
                data = cast(dict[str, Any], msgspec.json.decode(val))
                return OptionGreeks(**data)
            return None
        except Exception as e:
            logger.error("cache_get_greeks_failed", error=str(e), key=key)
            return None

    async def set_greeks(
        self,
        params: "BSParameters",
        option_type: str,
        greeks: "OptionGreeks",
        ttl: int = 3600,
    ) -> bool:
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for multi-layer caching with OPTIMIZED X-Fetch (Probabilistic Early Recomputation).
    Layer 1: Local In-Memory LRU
    Layer 2: Distributed Redis
    """
    l1_cache: TTLCache[str, object] = TTLCache(maxsize=maxsize, ttl=ttl)
    # L2 Probabilistic Early Recomputation (X-Fetch) Calibration:
    # beta: 1.0 (Standard). Increase for more aggressive refresh before TTL expiry.
    # delta_ms: Estimated computation time for 'func'. Default: 100ms.
    beta = 1.0
    delta_ms = 100

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            import math
            import random

            key_params = kwargs.copy()
            for i, arg in enumerate(args[1:]):
                key_params[f"arg_{i}"] = arg

            cache_key = generate_cache_key(prefix, **key_params)

            # 1. L1 Check (with X-Fetch logic for local memory?)
            if cache_key in l1_cache:
                return l1_cache[cache_key]

            # 2. L2 Check (Redis) with X-Fetch implementation
            redis = get_redis()
            cached_val = None

            if redis:
                try:
                    pipe = redis.pipeline()
                    pipe.get(cache_key)
                    pipe.pttl(cache_key)
                    results = await pipe.execute()
                    cached_val, remaining_ms = results[0], results[1]

                    if cached_val:
                        delta_ms = 100  # Assume 100ms computation time average
                        if (
                            remaining_ms > 0
                            and (remaining_ms - delta_ms * beta * math.log(random.random())) < 0  # nosec B311
                        ):
                            logger.info("x_fetch_triggered_early_refresh", key=cache_key)
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
                    await redis.setex(cache_key, 3600, msgspec.json.encode(result))
                except Exception as e:
                    logger.warning("l2_cache_write_failed", error=str(e))

            return result

        return wrapper

    return decorator


class RateLimitTier(StrEnum):
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
        # 1. Primary: Redis (Fastest)
        redis = get_redis()
        if redis:
            # Convert string to Enum if needed
            tier_enum = RateLimitTier.FREE
            if isinstance(tier, str):
                try:
                    tier_enum = RateLimitTier(tier.lower())
                except ValueError:
                    tier_enum = RateLimitTier.FREE
            else:
                tier_enum = tier

            config = RATE_LIMIT_CONFIGS[tier_enum]
            limit = (
                config.pricing_requests_per_minute
                if "price" in endpoint.lower()
                else config.requests_per_minute
            )
            key = f"rl:{user_id}:{endpoint}"
            now_ts = int(time.time())
            window = now_ts // 60
            full_key = f"{key}:{window}"
            try:
                pipe = redis.pipeline()
                pipe.incr(full_key)
                pipe.expire(full_key, 120)
                results = await pipe.execute()
                if results[0] <= (limit + config.burst_size):
                    return True
            except Exception as e:
                logger.warning(
                    "redis_rate_limit_check_failed_falling_back", error=str(e), user_id=user_id
                )

        # 2. Secondary: Optimized Postgres Unlogged Table (Highly Robust)
        try:
            from datetime import UTC, datetime

            from sqlalchemy import text

            from src.database import get_async_db_context

            # Simple window-based check mirroring Redis logic
            now_dt = datetime.now(UTC)
            window_start = now_dt.replace(second=0, microsecond=0)

            async with get_async_db_context() as db:
                # Optimized native query for unlogged rate_limits table
                stmt = text("""
                    INSERT INTO rate_limits (user_id, endpoint, window_start, request_count)
                    VALUES (:uid, :ep, :ws, 1)
                    ON CONFLICT (user_id, endpoint, window_start) 
                    DO UPDATE SET request_count = rate_limits.request_count + 1
                    RETURNING request_count;
                """)
                result = await db.execute(
                    stmt, {"uid": user_id, "ep": endpoint, "ws": window_start}
                )
                count = result.scalar()
                await db.commit()

                # Use default free tier limits for DB fallback safety if config not resolved
                db_limit = 100
                tier_enum = RateLimitTier.FREE
                if isinstance(tier, str):
                    try:
                        tier_enum = RateLimitTier(tier.lower())
                    except ValueError:
                        tier_enum = RateLimitTier.FREE
                else:
                    tier_enum = tier

                config = RATE_LIMIT_CONFIGS[tier_enum]
                db_limit = (
                    config.pricing_requests_per_minute
                    if "price" in endpoint.lower()
                    else config.requests_per_minute
                )

                return bool(cast(int, count) <= db_limit)

        except Exception as e:
            logger.error("db_rate_limit_fallback_failed", error=str(e), user_id=user_id)
            return True  # Fail open


rate_limiter = RateLimiter()


async def warm_cache() -> None:
    """Pre-warm cache with common option parameters in parallel."""
    from src.shared.pricing.black_scholes import BlackScholesEngine

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
                    from src.math_kernel.models import BSParameters

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
                        pricing_cache.set_option_price(params, "call", "bs_unified", float(price))
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

    async def get_user(self, user_id: str) -> dict[str, object] | None:
        redis = get_redis()
        if redis is None:
            return None
        try:
            val = await redis.get(f"{self.PREFIX}user:{user_id}")
            return cast(dict[str, Any], msgspec.json.decode(val)) if val else None
        except Exception as e:
            logger.error("db_cache_get_user_failed", error=str(e), user_id=user_id)
            return None

    async def set_user(self, user_id: str, user_data: dict[str, object], ttl: int = 300) -> bool:
        redis = get_redis()
        if redis is None:
            return False
        try:
            await redis.setex(f"{self.PREFIX}user:{user_id}", ttl, msgspec.json.encode(user_data))
            return True
        except Exception as e:
            logger.error("db_cache_set_user_failed", error=str(e), user_id=user_id)
            return False

    async def get_api_key(self, key_hash: str) -> dict[str, Any] | None:
        """Retrieve cached API key response data."""
        redis = get_redis()
        if redis is None:
            return None
        try:
            val = await redis.get(f"{self.PREFIX}api_key:{key_hash}")
            return cast(dict[str, Any], msgspec.json.decode(val)) if val else None
        except Exception as e:
            logger.error("db_cache_get_api_key_failed", error=str(e), key_hash=key_hash[:10])
            return None

    async def set_api_key(self, key_hash: str, key_data: dict[str, Any], ttl: int = 600) -> bool:
        """Cache API key response data."""
        redis = get_redis()
        if redis is None:
            return False
        try:
            await redis.setex(
                f"{self.PREFIX}api_key:{key_hash}", ttl, msgspec.json.encode(key_data)
            )
            return True
        except Exception as e:
            logger.error("db_cache_set_api_key_failed", error=str(e), key_hash=key_hash[:10])
            return False


db_cache = DatabaseQueryCache()

# --- Real-time updates support ---
redis_channel_updates: str = "pricing_updates"


async def publish_to_redis(channel: str, message: dict[str, Any]) -> None:
    """Publish a message to a Redis channel using msgspec."""
    redis = get_redis()
    if redis is not None:
        try:
            encoded_message = msgspec.json.encode(message)
            await redis.publish(channel, encoded_message)
            logger.debug("redis_publish_success", channel=channel)
        except Exception as e:
            logger.error("redis_publish_failed", error=str(e), channel=channel)


class RedisStreamManager:
    """
    High-Performance Redis Streams Manager.
    Supports Consumer Groups and ACK-based delivery guarantees.
    """

    @staticmethod
    async def xadd(stream: str, fields: dict[str, Any], maxlen: int = 10000) -> str | None:
        """Append to a Redis Stream with ultra-fast msgspec encoding."""
        redis = get_redis()
        if not redis:
            return None
        try:
            # Flatten dict for XADD (Redis streams require flat key-value pairs)
            flat_fields = {
                k: msgspec.json.encode(v) if not isinstance(v, (str, bytes, int, float)) else v
                for k, v in fields.items()
            }
            return await redis.xadd(stream, flat_fields, maxlen=maxlen, approximate=True)
        except Exception as e:
            logger.error("redis_xadd_failed", error=str(e), stream=stream)
            return None

    @staticmethod
    async def create_consumer_group(stream: str, group: str, start_id: str = "$") -> bool:
        """Create a Redis consumer group for horizontal scaling."""
        redis = get_redis()
        if not redis:
            return False
        try:
            await redis.xgroup_create(stream, group, id=start_id, mkstream=True)
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                return True
            logger.error("redis_xgroup_create_failed", error=str(e), stream=stream)
            return False

    @staticmethod
    async def xread_group(
        stream: str, group: str, consumer: str, count: int = 10, block_ms: int = 1000
    ) -> list[Any]:
        """Read from a Redis stream as part of a consumer group."""
        redis = get_redis()
        if not redis:
            return []
        try:
            streams = {stream: ">"}
            response = await redis.xreadgroup(group, consumer, streams, count=count, block=block_ms)
            return response
        except Exception as e:
            logger.error("redis_xreadgroup_failed", error=str(e), stream=stream)
            return []

    @staticmethod
    async def xack(stream: str, group: str, *ids: str) -> int:
        """Acknowledge processed stream messages."""
        redis = get_redis()
        if not redis:
            return 0
        try:
            return await redis.xack(stream, group, *ids)
        except Exception as e:
            logger.error("redis_xack_failed", error=str(e), stream=stream)
            return 0


stream_manager = RedisStreamManager()


async def get_redis_client() -> Redis:
    """FastAPI dependency to get the Redis client."""
    redis = get_redis()
    if redis is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Redis client not initialized")
    return redis


async def init_redis_cache(**kwargs: Any) -> None:
    """Initialize the Redis cache during startup."""
    import os

    if os.getenv("BSOPT_ALLOW_WEAK_SECRETS") == "1":
        logger.info("redis_cache_initialized_mock", details="Bypassed via BSOPT_ALLOW_WEAK_SECRETS")
        return

    redis = get_redis()
    if redis:
        try:
            await redis.ping()
            logger.info("redis_cache_initialized")
        except Exception as e:
            logger.error("redis_cache_init_failed", error=str(e))


async def close_redis_cache() -> None:
    """Close the Redis client connection."""
    global _redis
    if _redis:
        try:
            await _redis.aclose()
            _redis = None
            logger.info("redis_cache_closed")
        except Exception as e:
            logger.error("redis_cache_close_failed", error=str(e))


def generate_consistent_key(request: Any, prefix: str) -> str:
    """Generate a high-performance consistent cache key for HTTP requests."""

    params = sorted(request.query_params.items())

    # Use blake2b for faster hashing than sha256
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(request.url.path.encode())
    for k, v in params:
        hasher.update(f"{k}:{v}".encode())

    return f"{prefix}:{hasher.hexdigest()}"


def cached_endpoint(
    prefix: str = "api_cache", ttl: int = 60
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for FastAPI endpoints with Response-aware caching.
    """
    from fastapi import Request, Response

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = next((arg for arg in args if isinstance(arg, Request)), None) or kwargs.get(
                "request"
            )

            if not request:
                return await func(*args, **kwargs)

            redis = get_redis()
            if not redis:
                return await func(*args, **kwargs)

            cache_key = generate_consistent_key(request, prefix)

            try:
                cached = await redis.get(cache_key)
                if cached:
                    return Response(
                        content=cached,
                        media_type="application/json",
                        headers={"X-Cache": "HIT"},
                    )
            except Exception:
                pass

            response = await func(*args, **kwargs)

            # 3. Intelligent Caching
            try:
                data = None
                if isinstance(response, Response):
                    data = response.body
                elif hasattr(response, "model_dump_json"):
                    data = response.model_dump_json().encode()
                elif isinstance(response, dict | list):
                    data = msgspec.json.encode(response)

                if data:
                    await redis.setex(cache_key, ttl, data)
            except Exception as e:
                logger.error("api_cache_write_error", error=str(e))

            return response

        return wrapper

    return decorator
