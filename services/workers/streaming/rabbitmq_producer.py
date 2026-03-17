import aio_pika
import msgspec
import structlog

from core.shared.config import settings

from services.pricing.base import Producer

logger = structlog.get_logger(__name__)


class RabbitMQMarketDataProducer(Producer):
    """
    High-performance RabbitMQ producer for real-time market data.
    Decouples scrapers from the database for asynchronous persistence.
    """

    def __init__(self, broker_url: str | None = None):
        self.broker_url = broker_url or settings.RABBITMQ_URL
        self._connection = None
        self._channel = None
        self._exchange = None
        self._encoder = msgspec.json.Encoder()

    async def connect(self):
        """Establish a robust connection to RabbitMQ."""
        if self._connection is None:
            self._connection = await aio_pika.connect_robust(self.broker_url)
            self._channel = await self._connection.channel()

            # Declare the exchange for market data
            self._exchange = await self._channel.declare_exchange(
                "market_data", aio_pika.ExchangeType.TOPIC, durable=True
            )

            # Declare the main queue with a Dead Letter Exchange
            await self._channel.declare_queue(
                "market_ticks_persistence",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "market_data_dlx",
                    "x-dead-letter-routing-key": "dead_ticks",
                },
            )

            # Declare the DLX
            dlx = await self._channel.declare_exchange(
                "market_data_dlx", aio_pika.ExchangeType.DIRECT, durable=True
            )
            dlq = await self._channel.declare_queue("market_data_dlq", durable=True)
            await dlq.bind(dlx, routing_key="dead_ticks")

            logger.info("rabbitmq_producer_connected", broker=self.broker_url)

    async def produce_market_data(self, data: dict, routing_key: str = "nse.ticks"):
        """
        Publish market data to the topic exchange.
        """
        if not self._exchange:
            await self.connect()

        try:
            message_body = self._encoder.encode(data)
            message = aio_pika.Message(
                body=message_body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            )

            await self._exchange.publish(message, routing_key=routing_key)
            logger.debug("rabbitmq_message_published", routing_key=routing_key)
        except Exception as e:
            logger.error("rabbitmq_publish_failed", error=str(e))
            raise

    async def close(self):
        """Gracefully close the connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("rabbitmq_producer_closed")
