import asyncio
import os
from collections.abc import Callable
from typing import Any

import aio_pika
import structlog

logger = structlog.get_logger()


class MessageBroker:
    """
    Asynchronous Message Broker (RabbitMQ/aio-pika).
    Handles task distribution and inter-service signaling.
    """

    def __init__(self):
        self.url = os.getenv("RABBITMQ_URL", "amqp://bsopt_admin:bsopt_rmq_secret@rabbitmq:5672/")
        self._connection: aio_pika.RobustConnection | None = None
        self._channel: aio_pika.RobustChannel | None = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Establish a robust connection to RabbitMQ with locking."""
        if self._connection and not self._connection.is_closed:
            return

        async with self._lock:
            if not self._connection or self._connection.is_closed:
                self._connection = await aio_pika.connect_robust(
                    self.url,
                    timeout=10,
                    client_properties={"connection_name": "bsopt-api"}
                )
                self._channel = await self._connection.channel()
                logger.info("rabbitmq_connected", url=self.url.split("@")[-1])

    async def health_check(self) -> dict[str, Any]:
        """Check RabbitMQ connectivity health."""
        try:
            if not self._connection or self._connection.is_closed:
                await asyncio.wait_for(self.connect(), timeout=5.0)
            
            # Ping by declaring a temporary passive queue or just checking channel
            if self._channel and not self._channel.is_closed:
                return {"status": "healthy", "url": self.url.split("@")[-1]}
            return {"status": "unhealthy", "reason": "channel_closed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def publish(self, queue_name: str, message: bytes, priority: int = 0):
        """Publish a message to a specific queue."""
        await self.connect()
        if self._channel:
            await self._channel.default_exchange.publish(
                aio_pika.Message(body=message, priority=priority), routing_key=queue_name
            )
            logger.debug("message_published", queue=queue_name)

    async def consume(self, queue_name: str, callback: Callable[[aio_pika.IncomingMessage], Any]):
        """Subscribe to a queue and process messages."""
        await self.connect()
        if self._channel:
            queue = await self._channel.declare_queue(queue_name, durable=True)
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        await callback(message)

    async def close(self):
        """Gracefully close the connection."""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None
                self._channel = None
                logger.info("rabbitmq_connection_closed")


broker = MessageBroker()
