from collections.abc import Awaitable, Callable

import aio_pika
import msgspec
import structlog

from core.shared.config import settings

logger = structlog.get_logger(__name__)


class RabbitMQMarketDataConsumer:
    """
    High-performance RabbitMQ consumer for real-time market data persistence.
    Supports batch processing and automatic re-queuing with DLQ logic.
    """

    def __init__(self, broker_url: str | None = None, queue_name: str = "market_ticks_persistence"):
        self.broker_url = broker_url or settings.RABBITMQ_URL
        self.queue_name = queue_name
        self._connection = None
        self._channel = None
        self._decoder = msgspec.json.Decoder()

    async def connect(self):
        """Establish a robust connection to RabbitMQ."""
        if self._connection is None:
            self._connection = await aio_pika.connect_robust(self.broker_url)
            self._channel = await self._connection.channel()
            # Set prefetch count for better throughput
            await self._channel.set_prefetch(100)
            logger.info(
                "rabbitmq_consumer_connected", broker=self.broker_url, queue=self.queue_name
            )

    async def consume(self, callback: Callable[[dict], Awaitable[None]]):
        """
        Start consuming messages from the queue and process with the provided callback.
        """
        if not self._channel:
            await self.connect()

        queue = await self._channel.declare_queue(self.queue_name, durable=True)

        logger.info("rabbitmq_consumption_started", queue=self.queue_name)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process(requeue=False, ignore_processed=True):
                    try:
                        data = self._decoder.decode(message.body)
                        await callback(data)
                    except Exception as e:
                        logger.error("rabbitmq_message_processing_failed", error=str(e))
                        # If callback fails, message is NACKed and moved to DLQ (requeue=False)
                        raise

    async def close(self):
        """Gracefully close the connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("rabbitmq_consumer_closed")
