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
        if not redis:
            return False
        return bool(await redis.exists(f"blacklist:{jti}"))

    async def contains(self, jti: str) -> bool:
        """Alias for is_token_revoked."""
        return await self.is_token_revoked(jti)

    async def revoke_token(self, token_data: TokenData) -> None:
        """Revoke a token by adding its JTI to the blacklist."""
        await self.add(token_data.jti, token_data.exp)

    async def add(self, jti: str | None, expires_at: datetime) -> None:
        """Revoke a token by adding its JTI to the blacklist (Universal)."""
        if not jti:
            return

        try:
            redis = await get_redis_client()
            if not redis:
                return

            now = datetime.now(UTC)
            # Ensure expires_at is aware if it's not
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)

            ttl = int((expires_at - now).total_seconds())
            if ttl > 0:
                await redis.setex(f"blacklist:{jti}", ttl, "1")
                logger.info("token_revoked_successfully", jti=jti, ttl=ttl)
        except Exception as e:
            logger.warning(f"token_revocation_failed: {e}", jti=jti)

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