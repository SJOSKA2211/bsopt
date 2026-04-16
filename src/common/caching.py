from datetime import UTC, datetime, timedelta
from typing import Any

from cachetools import TTLCache

from src.shared.utils.cache import db_cache, get_redis_client

# Define constants for cache TTLs
USER_LOCAL_TTL = 60  # 1 minute for local cache
API_KEY_LOCAL_TTL = 60  # 1 minute for local cache
USER_REDIS_TTL = 300 # 5 minutes for Redis cache
API_KEY_REDIS_TTL = 300 # 5 minutes for Redis cache


class CentralizedCacheService:
    """
    Provides a unified interface for local and distributed caching of users and API keys.
    Uses TTLCache for local in-memory caching and a distributed cache (e.g., Redis via db_cache)
    for cross-process consistency.
    """

    def __init__(self):
        self._user_local_cache = TTLCache(maxsize=10000, ttl=USER_LOCAL_TTL)
        self._api_key_local_cache = TTLCache(maxsize=10000, ttl=API_KEY_LOCAL_TTL)

    async def get_user_cached(self, user_id: str) -> dict[str, Any] | None:
        """
        Retrieves user data, first from local cache, then from distributed cache.
        Returns None if not found in either.
        """
        # 1. Try local cache first
        if user_id in self._user_local_cache:
            return self._user_local_cache[user_id]

        # 2. Try distributed cache
        try:
            cached_data = await db_cache.get_user(user_id)
            if cached_data:
                # Populate local cache from distributed cache
                self._user_local_cache[user_id] = cached_data
                return cached_data
        except Exception as e:
            # Log error but continue, as local cache might still be available or DB can be used
            print(f"Error fetching user {user_id} from distributed cache: {e}")
            pass
        
        return None

    async def set_user_cached(self, user_id: str, user_data: dict[str, Any]):
        """
        Stores user data in both local and distributed caches.
        """
        # Store in local cache
        self._user_local_cache[user_id] = user_data
        
        # Store in distributed cache with a specific TTL
        try:
            await db_cache.set_user(user_id, user_data, ttl=timedelta(seconds=USER_REDIS_TTL))
        except Exception as e:
            print(f"Error setting user {user_id} in distributed cache: {e}")
            pass

    async def get_api_key_cached(self, key_hash: str) -> dict[str, Any] | None:
        """
        Retrieves API key data, first from local cache, then from distributed cache.
        Returns None if not found in either.
        """
        # 1. Try local cache first
        if key_hash in self._api_key_local_cache:
            return self._api_key_local_cache[key_hash]

        # 2. Try distributed cache
        try:
            cached_data = await db_cache.get_api_key(key_hash)
            if cached_data:
                # Populate local cache from distributed cache
                self._api_key_local_cache[key_hash] = cached_data
                return cached_data
        except Exception as e:
            print(f"Error fetching API key {key_hash[:10]}... from distributed cache: {e}")
            pass
        
        return None

    async def set_api_key_cached(self, key_hash: str, api_key_data: dict[str, Any]):
        """
        Stores API key data in both local and distributed caches.
        """
        # Store in local cache
        self._api_key_local_cache[key_hash] = api_key_data
        
        # Store in distributed cache with a specific TTL
        try:
            await db_cache.set_api_key(key_hash, api_key_data, ttl=timedelta(seconds=API_KEY_REDIS_TTL))
        except Exception as e:
            print(f"Error setting API key {key_hash[:10]}... in distributed cache: {e}")
            pass

    async def update_api_key_last_used(self, key_hash: str):
        """
        Updates the last_used_at timestamp for an API key in the distributed cache.
        """
        try:
            redis = await get_redis_client()
            if redis:
                await redis.hset("api_key_last_used", key_hash, datetime.now(UTC).isoformat())
        except Exception as e:
            print(f"Error updating API key last used time for {key_hash[:10]}...: {e}")
            pass

    async def get_token_data_cached(self, token: str) -> Any | None:
        """
        Retrieves token data from distributed cache.
        """
        try:
            return await db_cache.get(f"token:{token}")
        except Exception:
            return None

    async def set_token_data_cached(self, token: str, token_data: Any):
        """
        Stores token data in distributed cache.
        """
        try:
            # Token data usually has an expiration, but for simplicity we use a default
            await db_cache.set(f"token:{token}", token_data, ttl=timedelta(minutes=30))
        except Exception:
            pass

    async def revoke_token_cached(self, token: str):
        """
        Removes token data from distributed cache upon revocation.
        """
        try:
            await db_cache.delete(f"token:{token}")
        except Exception:
            pass


# Instantiate the cache service
centralized_cache_service = CentralizedCacheService()
