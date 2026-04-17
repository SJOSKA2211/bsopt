from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from cachetools import TTLCache

from src.shared.utils.cache import db_cache, get_redis_client

logger = structlog.get_logger(__name__)

USER_LOCAL_TTL = 60
API_KEY_LOCAL_TTL = 60
USER_REDIS_TTL = 300
API_KEY_REDIS_TTL = 300


class CentralizedCacheService:
    """
    Unified interface for local and distributed caching.
    """

    def __init__(self):
        self._user_local_cache = TTLCache(maxsize=10000, ttl=USER_LOCAL_TTL)
        self._api_key_local_cache = TTLCache(maxsize=10000, ttl=API_KEY_LOCAL_TTL)

    async def get_user_cached(self, user_id: str) -> dict[str, Any] | None:
        # 1. Local cache
        if user_id in self._user_local_cache:
            return self._user_local_cache[user_id]

        # 2. Distributed cache
        try:
            cached_data = await db_cache.get_user(user_id)
            if cached_data:
                self._user_local_cache[user_id] = cached_data
                return cached_data
        except Exception as e:
            logger.warning("distributed_cache_get_user_failed", user_id=user_id, error=str(e))
        
        return None

    async def set_user_cached(self, user_id: str, user_data: dict[str, Any]):
        self._user_local_cache[user_id] = user_data
        try:
            await db_cache.set_user(user_id, user_data, ttl=timedelta(seconds=USER_REDIS_TTL))
        except Exception as e:
            logger.error("distributed_cache_set_user_failed", user_id=user_id, error=str(e))

    async def get_api_key_cached(self, key_hash: str) -> dict[str, Any] | None:
        # 1. Local cache
        if key_hash in self._api_key_local_cache:
            return self._api_key_local_cache[key_hash]

        # 2. Distributed cache
        try:
            cached_data = await db_cache.get_api_key(key_hash)
            if cached_data:
                self._api_key_local_cache[key_hash] = cached_data
                return cached_data
        except Exception as e:
            logger.warning("distributed_cache_get_api_key_failed", error=str(e))
        
        return None

    async def set_api_key_cached(self, key_hash: str, api_key_data: dict[str, Any]):
        self._api_key_local_cache[key_hash] = api_key_data
        try:
            await db_cache.set_api_key(key_hash, api_key_data, ttl=timedelta(seconds=API_KEY_REDIS_TTL))
        except Exception as e:
            logger.error("distributed_cache_set_api_key_failed", error=str(e))

    async def update_api_key_last_used(self, key_hash: str):
        try:
            redis = await get_redis_client()
            if redis:
                await redis.hset("api_key_last_used", key_hash, datetime.now(UTC).isoformat())
        except Exception as e:
            logger.error("update_api_key_last_used_failed", error=str(e))

    async def get_token_data_cached(self, token: str) -> Any | None:
        try:
            return await db_cache.get(f"token:{token}")
        except Exception as e:
            logger.warning("get_token_data_cached_failed", error=str(e))
            return None

    async def set_token_data_cached(self, token: str, token_data: Any):
        try:
            await db_cache.set(f"token:{token}", token_data, ttl=timedelta(minutes=30))
        except Exception as e:
            logger.error("set_token_data_cached_failed", error=str(e))

    async def revoke_token_cached(self, token: str):
        try:
            await db_cache.delete(f"token:{token}")
        except Exception as e:
            logger.error("revoke_token_cached_failed", error=str(e))


centralized_cache_service = CentralizedCacheService()