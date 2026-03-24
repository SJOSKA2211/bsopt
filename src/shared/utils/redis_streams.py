"""
Redis Streams Utility (High-Performance)

Handles XADD, XREADGROUP, and Consumer Group management for EquaFlow.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import orjson
import structlog
from redis.asyncio import Redis

from src.shared.utils.cache import get_redis_client

logger = structlog.get_logger(__name__)

class RedisStreamManager:
    """
    Manages Redis Streams for real-time data and internal events.
    """

    def __init__(self, stream_name: str, group_name: str = "equaflow_consumers"):
        self.stream_name = stream_name
        self.group_name = group_name
        self._redis: Redis | None = None

    async def _get_redis(self) -> Redis:
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    async def setup_group(self) -> None:
        """Create the consumer group if it doesn't exist."""
        r = await self._get_redis()
        try:
            await r.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
            logger.info(
                "redis_stream_group_created", stream=self.stream_name, group=self.group_name
            )
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.debug(
                    "redis_stream_group_exists", stream=self.stream_name, group=self.group_name
                )
            else:
                logger.error("redis_stream_group_setup_failed", error=str(e))

    async def publish(self, data: dict[str, Any], max_len: int = 100000) -> str:
        """
        Publish data to the stream (XADD).
        Automatically prunes the stream to max_len.
        """
        r = await self._get_redis()
        try:
            # Flatten dict for Redis Stream (only supports field-value pairs)
            # We'll use a single field 'payload' with JSON string
            payload = orjson.dumps(data).decode("utf-8")
            msg_id = await r.xadd(
                self.stream_name, {"payload": payload}, maxlen=max_len, approximate=True
            )
            return msg_id
        except Exception as e:
            logger.error("redis_stream_publish_failed", stream=self.stream_name, error=str(e))
            raise

    async def consume(
        self,
        consumer_name: str,
        handler: Callable[[str, dict[str, Any]], Any],
        batch_size: int = 10,
        block_ms: int = 5000,
    ) -> None:
        """
        Consume messages using XREADGROUP.
        Acknowledges messages after successful processing.
        """
        r = await self._get_redis()
        await self.setup_group()

        logger.info(
            "redis_stream_consumer_started", stream=self.stream_name, consumer=consumer_name
        )

        while True:
            try:
                # 1. Read new messages (ID '>')
                streams = {self.stream_name: ">"}
                messages = await r.xreadgroup(
                    self.group_name, consumer_name, streams, count=batch_size, block=block_ms
                )

                if not messages:
                    continue

                for stream, msgs in messages:
                    for msg_id, payload in msgs:
                        try:
                            data = orjson.loads(payload[b"payload"])
                            # 2. Process message
                            await handler(msg_id.decode(), data)
                            # 3. Acknowledge message (XACK)
                            await r.xack(self.stream_name, self.group_name, msg_id)
                        except Exception as inner_e:
                            logger.error(
                                "redis_stream_handler_error", msg_id=msg_id, error=str(inner_e)
                            )
                            # Depending on strategy: retry, DLQ, or drop

            except asyncio.CancelledError:
                logger.info("redis_stream_consumer_cancelled", stream=self.stream_name)
                break
            except Exception as e:
                logger.error("redis_stream_consume_loop_failed", error=str(e))
                await asyncio.sleep(1)  # Backoff on connection error
