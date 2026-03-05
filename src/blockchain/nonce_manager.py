import asyncio
import structlog
from redis.asyncio import Redis
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)

class NonceManager:
    """
    Atomic Nonce Manager using Redis.
    Ensures cross-process nonce synchronization for high-concurrency wallet operations.
    """
    def __init__(self, address: str, chain_id: int):
        self.address = address
        self.chain_id = chain_id
        self.redis_key = f"nonce:{chain_id}:{address}"
        self._lock = asyncio.Lock()

    async def get_next_nonce(self, w3_nonce_func) -> int:
        """
        Get the next available nonce, syncing with the chain if necessary.
        """
        redis: Redis = get_redis()
        if not redis:
            # Fallback to local if redis is down (not ideal for multi-process)
            return await w3_nonce_func()

        async with self._lock:
            # Try to get from Redis
            nonce_bytes = await redis.get(self.redis_key)
            
            if nonce_bytes is None:
                # Sync with chain
                nonce = await w3_nonce_func()
                await redis.set(self.redis_key, nonce)
                return nonce
            
            # Increment and return
            new_nonce = await redis.incr(self.redis_key)
            return int(new_nonce) - 1

    async def reset(self, w3_nonce_func):
        """Force sync with chain."""
        redis: Redis = get_redis()
        if redis:
            nonce = await w3_nonce_func()
            await redis.set(self.redis_key, nonce)
            logger.info("nonce_reset", address=self.address, nonce=nonce)
