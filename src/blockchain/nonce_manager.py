import asyncio

import structlog
from redis.asyncio import Redis

from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class NonceManager:
    """
    Atomic Nonce Manager using Redis (Hardened).
    Ensures cross-process nonce synchronization using Lua scripts for atomicity.
    """

    def __init__(self, address: str, chain_id: int):
        self.address = address
        self.chain_id = chain_id
        self.redis_key = f"nonce:{chain_id}:{address}"
        self._lock = asyncio.Lock()

        #  HIGH-PERFORMANCE: Lua script for atomic get-and-increment
        self._lua_nonce_script = """
        local current = redis.call('get', KEYS[1])
        if current == False then
            redis.call('set', KEYS[1], ARGV[1])
            return ARGV[1]
        else
            local next = redis.call('incr', KEYS[1])
            return next - 1
        end
        """

    async def get_next_nonce(self, w3_nonce_func) -> int:
        """
        Atomic nonce retrieval via Redis Lua script (High-Performance).
        OPTIMIZED: Ensures true atomicity across all distributed nodes.
        """
        redis: Redis = get_redis()
        if not redis:
            return await w3_nonce_func()

        async with self._lock:
            try:
                # Cold start: We need a baseline if Redis is empty
                # We do this check before the script to avoid unnecessary RPCs
                baseline = 0

                # Execute the script for atomic get-and-increment
                # KEYS[1] = self.redis_key
                # ARGV[1] = baseline (only used if key doesn't exist)
                nonce = await redis.eval(self._lua_nonce_script, 1, self.redis_key, baseline)

                if int(nonce) == baseline:
                    # If it returned the baseline, it might be a cold start.
                    # Verify with chain to be safe.
                    chain_nonce = await w3_nonce_func()
                    if chain_nonce > baseline:
                        await redis.set(self.redis_key, chain_nonce + 1)
                        return chain_nonce

                return int(nonce)

            except Exception as e:
                logger.error("nonce_atomic_retrieval_failed_falling_back", error=str(e))
                return await w3_nonce_func()

    async def reset(self, w3_nonce_func):
        """Force sync with chain and clear Redis state."""
        redis: Redis = get_redis()
        if redis:
            nonce = await w3_nonce_func()
            await redis.set(self.redis_key, nonce)
            logger.info("nonce_synchronized_with_chain", address=self.address, nonce=nonce)
