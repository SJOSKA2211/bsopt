import asyncio
import os
from collections.abc import Callable
from typing import Any

import aio_pika
import structlog

logger = structlog.get_logger()


class MessageBroker:
    """
    Optimized Asynchronous Message Broker (RabbitMQ/aio-pika).
    Implements connection pooling and robust error recovery.
    """

    def __init__(self):
        self.url = os.getenv("RABBITMQ_URL", "amqp://bsopt_admin:bsopt_rmq_secret@rabbitmq:5672/")
        self._connection_pool: aio_pika.pool.Pool | None = None
        self._channel_pool: aio_pika.pool.Pool | None = None
        self._lock = asyncio.Lock()

    async def _get_connection(self) -> aio_pika.RobustConnection:
        """Helper to create a robust connection for the pool."""
        return await aio_pika.connect_robust(
            self.url,
            timeout=10,
            client_properties={"connection_name": "bsopt-api-pool"}
        )

    async def _get_channel(self) -> aio_pika.Channel:
        """Helper to get a channel from the pool's connection."""
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
        async with self._connection_pool.acquire() as connection:
            return await connection.channel()

    async def connect(self):
        """Initialize connection and channel pools."""
        if self._connection_pool:
            return

        async with self._lock:
            if not self._connection_pool:
                self._connection_pool = aio_pika.pool.Pool(self._get_connection, max_size=2)
                self._channel_pool = aio_pika.pool.Pool(self._get_channel, max_size=10)
                logger.info("rabbitmq_pools_initialized", url=self.url.split("@")[-1])

    async def health_check(self) -> dict[str, Any]:
        """Check RabbitMQ connectivity health via pool acquisition."""
        try:
            await self.connect()
            async with self._channel_pool.acquire() as channel:
                if not channel.is_closed:
                    return {"status": "healthy", "url": self.url.split("@")[-1], "type": "pooled"}
            return {"status": "unhealthy", "reason": "pool_exhausted_or_closed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def publish(self, queue_name: str, message: bytes, priority: int = 0, durable: bool = True):
        """Publish a message using a pooled channel."""
        await self.connect()
        async with self._channel_pool.acquire() as channel:
            # Using the default exchange
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=message,
                    priority=priority,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT if durable else aio_pika.DeliveryMode.NOT_PERSISTENT
                ),
                routing_key=queue_name
            )
            logger.debug("message_published_pooled", queue=queue_name)

    async def consume(self, queue_name: str, callback: Callable[[aio_pika.IncomingMessage], Any], prefetch: int = 10):
        """Subscribe to a queue and process messages with high-throughput prefetching."""
        await self.connect()
        async with self._channel_pool.acquire() as channel:
            await channel.set_qos(prefetch_count=prefetch)
            queue = await channel.declare_queue(queue_name, durable=True)
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            await callback(message)
                        except Exception as e:
                            logger.error("message_processing_failed", error=str(e), queue=queue_name)

    async def close(self):
        """Gracefully close all pools."""
        async with self._lock:
            if self._channel_pool:
                await self._channel_pool.close()
            if self._connection_pool:
                await self._connection_pool.close()
            self._connection_pool = None
            self._channel_pool = None
            logger.info("rabbitmq_pools_closed")


broker = MessageBroker()
