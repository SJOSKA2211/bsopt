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
        self.connection: aio_pika.RobustConnection | None = None
        self.channel: aio_pika.RobustChannel | None = None

    async def connect(self):
        """Establish a robust connection to RabbitMQ."""
        if not self.connection:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            logger.info("rabbitmq_connected")

    async def publish(self, queue_name: str, message: bytes, priority: int = 0):
        """Publish a message to a specific queue."""
        await self.connect()
        await self.channel.default_exchange.publish(
            aio_pika.Message(body=message, priority=priority), routing_key=queue_name
        )
        logger.debug("message_published", queue=queue_name)

    async def consume(self, queue_name: str, callback: Callable[[aio_pika.IncomingMessage], Any]):
        """Subscribe to a queue and process messages."""
        await self.connect()
        queue = await self.channel.declare_queue(queue_name, durable=True)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    await callback(message)

    async def close(self):
        """Gracefully close the connection."""
        if self.connection:
            await self.connection.close()
            logger.info("rabbitmq_connection_closed")

broker = MessageBroker()
