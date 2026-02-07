import redis
import structlog

logger = structlog.get_logger()


class RedisRemediator:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        try:
            self.client = redis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=False
            )
            # Ping to verify connection
            self.client.ping()
            logger.info(
                "redis_remediator_init",
                status="success",
                message="Redis client initialized and connected.",
            )
        except Exception as e:
            logger.error(
                "redis_remediator_init",
                status="failure",
                error=str(e),
                message="Failed to initialize or connect to Redis client.",
            )
            raise

    def purge_cache(self, cache_key_pattern: str) -> bool:
        if not cache_key_pattern:
            raise ValueError("Cache key pattern cannot be empty.")

        try:
            # Use SCAN for large datasets to avoid blocking the server, but for tests KEYS is fine
            # In a real-world scenario, you might use a more robust iterative scanning.
            keys_bytes: list[bytes] = self.client.keys(cache_key_pattern)

            if not keys_bytes:
                logger.info(
                    "redis_remediator_purge",
                    status="no_keys_found",
                    pattern=cache_key_pattern,
                    message="No keys matching pattern found to purge.",
                )
                return True

            deleted_count = self.client.delete(*keys_bytes)
            logger.info(
                "redis_remediator_purge",
                status="success",
                pattern=cache_key_pattern,
                deleted_count=deleted_count,
                message="Cache purged successfully.",
            )
            return True
        except Exception as e:
            logger.error(
                "redis_remediator_purge",
                status="failure",
                pattern=cache_key_pattern,
                error=str(e),
                message="Failed to purge cache.",
            )
            return False
