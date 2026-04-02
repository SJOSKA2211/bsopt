"""
Session Management Substrate (Redis).
"""

import logging
import msgspec
import hashlib
from datetime import UTC, datetime
from src.shared.utils.cache import get_redis_client
from src.auth.core.tokens import TokenData

logger = logging.getLogger(__name__)

class SessionService:
    """
    Redis-based session tracking and token revocation.
    Optimized with hashed token keys for memory efficiency.
    """
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
            redis = await get_redis_client()
            ttl = int((token_data.exp - datetime.now(UTC)).total_seconds())
            if ttl > 0:
                hashed_key = self._hash_token(token)
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
            redis = await get_redis_client()
            hashed_key = self._hash_token(token)
            cached_data = await redis.get(f"session_v3:{hashed_key}")
            if cached_data:
                return msgspec.json.decode(cached_data, type=TokenData)
        except Exception:
            return None

# Global instance for easy access
session_service = SessionService()
