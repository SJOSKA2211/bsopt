"""
Session Management Substrate (Redis).
"""

import hashlib
import logging
from datetime import UTC, datetime

import msgspec
from cachetools import TTLCache

from src.auth.core.tokens import TokenData
from src.shared.utils.cache import get_redis_client

logger = logging.getLogger(__name__)


class SessionService:
    """
    Redis-based session tracking and token revocation.
    Optimized with hashed token keys for memory efficiency and local TTLCache for performance.
    """

    def __init__(self):
        # Local cache for session data to avoid Redis roundtrips for frequent validations
        self._local_session_cache = TTLCache(maxsize=10000, ttl=30)  # 30 seconds local TTL

    def _hash_token(self, token: str) -> str:
        """Deterministic hash of the token for use as Redis key."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token JTI is in the blacklist."""
        redis = await get_redis_client()
        return bool(await redis.exists(f"blacklist:{jti}"))

    async def revoke_token(self, token_data: TokenData) -> None:
        """Revoke a token by adding its JTI to the blacklist."""
        try:
            if token_data.jti:
                redis = await get_redis_client()
                ttl = int((token_data.exp - datetime.now(UTC)).total_seconds())
                if ttl > 0:
                    await redis.setex(f"blacklist:{token_data.jti}", ttl, "1")
        except Exception as e:
            logger.warning(f"token_revocation_failed: {e}")

    async def cache_session(self, token: str, token_data: TokenData) -> None:
        """Cache session data for fast path validation using hashed keys."""
        try:
            hashed_key = self._hash_token(token)
            # Update Local Cache
            self._local_session_cache[hashed_key] = token_data

            # Update Redis
            redis = await get_redis_client()
            ttl = int((token_data.exp - datetime.now(UTC)).total_seconds())
            if ttl > 0:
                await redis.setex(
                    f"session_v3:{hashed_key}",
                    ttl,
                    msgspec.json.encode(token_data),
                )
        except Exception as e:
            logger.warning("session_cache_write_failed", error=str(e))

    async def get_cached_session(self, token: str) -> TokenData | None:
        """Retrieve cached session data using hashed keys."""
        try:
            hashed_key = self._hash_token(token)

            # 1. Local Cache hit
            if hashed_key in self._local_session_cache:
                return self._local_session_cache[hashed_key]

            # 2. Redis hit
            redis = await get_redis_client()
            cached_data = await redis.get(f"session_v3:{hashed_key}")
            if cached_data:
                token_data = msgspec.json.decode(cached_data, type=TokenData)
                # Populate Local Cache
                self._local_session_cache[hashed_key] = token_data
                return token_data
        except Exception:
            return None


# Global instance for easy access
session_service = SessionService()
