import asyncio
from collections.abc import Callable
from typing import Any

import aio_pika
import msgspec
import structlog
from aio_pika.pool import Pool
from tenacity import retry, stop_after_attempt, wait_exponential

from src.shared.config import settings

logger = structlog.get_logger(__name__)


class RabbitMQManager:
    """
    Overhauled Production-Grade RabbitMQ Manager.
    Features:
    - Connection and Channel Pooling for high concurrency.
    - Ultra-fast serialization via msgspec.
    - Robust auto-reconnection and retry logic.
    - Standardized DLQ and Telemetry support.
    """

    def __init__(self):
        self.url = settings.RABBITMQ_URL
        self._connection_pool: Pool | None = None
        self._channel_pool: Pool | None = None
        self._lock = asyncio.Lock()

        # Configurable Topologies
        self.exchange_name = "market_data"
        self.queue_name = "market_ticks"
        self.dlq_name = "market_ticks_dlq"
        self.audit_exchange = "audit_exchange"
        self.audit_queue = "audit_logs"
        self.telemetry_exchange = "telemetry_exchange"
        self.telemetry_queue = "telemetry_logs"
        self.news_topic = "scraper.news"
        self.signal_topic = "model.signals"

        self._encoder = msgspec.json.Encoder()

    async def _get_connection(self) -> aio_pika.RobustConnection:
        """Establish a robust connection."""
        return await aio_pika.connect_robust(
            self.url,
            client_properties={"connection_name": "manifold-shared-manager"}
        )

    async def _get_channel(self) -> aio_pika.Channel:
        """Acquire a channel from the connection pool."""
        if not self._connection_pool:
            raise RuntimeError("RabbitMQ connection pool not initialized. Call connect() first.")
        async with self._connection_pool.acquire() as connection:
            return await connection.channel()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def connect(self):
        """Initialize connection and channel pools, and declare topologies."""
        async with self._lock:
            if self._connection_pool:
                return

            logger.info("initializing_rabbitmq_pools", url=self.url.split("@")[-1])
            self._connection_pool = Pool(self._get_connection, max_size=2)
            self._channel_pool = Pool(self._get_channel, max_size=20)

            # Infrastructure Bootstrap
            try:
                async with self._channel_pool.acquire() as channel:
                    # 1. Dead Letter Infrastructure
                    logger.info("declaring_dlq")
                    await channel.declare_queue(
                        self.dlq_name, 
                        durable=True,
                        arguments={"x-queue-mode": "lazy"}
                    )

                    # 2. Market Data Mesh
                    logger.info("declaring_market_ticks")
                    await channel.declare_queue(
                        self.queue_name,
                        durable=True,
                        arguments={
                            "x-dead-letter-exchange": "",
                            "x-dead-letter-routing-key": self.dlq_name,
                        },
                    )

                    # 3. Audit Substrate
                    logger.info("declaring_audit")
                    audit_queue = await channel.declare_queue(
                        self.audit_queue, 
                        durable=True,
                        arguments={"x-queue-mode": "lazy"}
                    )
                    audit_exchange = await channel.declare_exchange(self.audit_exchange, type="direct", durable=True)
                    await audit_queue.bind(audit_exchange, routing_key="audit")

                    # 4. Telemetry Hub
                    logger.info("declaring_telemetry")
                    telemetry_queue = await channel.declare_queue(self.telemetry_queue, durable=True)
                    telemetry_exchange = await channel.declare_exchange(self.telemetry_exchange, type="topic", durable=True)
                    await telemetry_queue.bind(telemetry_exchange, routing_key="telemetry.#")
                    # 5. Signal and News Streams
                    logger.info("declaring_signals_and_news")
                    await channel.declare_queue(self.news_topic, durable=True)
                    await channel.declare_queue(self.signal_topic, durable=True)

                    logger.info("rabbitmq_topology_confirmed")
                logger.info("rabbitmq_connect_complete")
            except Exception as e:
                logger.error("rabbitmq_bootstrap_failed", error=str(e))
                # Reset pools to allow retry
                self._connection_pool = None
                self._channel_pool = None
                raise

    async def _publish(self, exchange_name: str | None, routing_key: str, payload: Any):
        """Internal pooled publishing primitive."""
        if not self._channel_pool:
            await self.connect()

        async with self._channel_pool.acquire() as channel:
            message = aio_pika.Message(
                body=self._encoder.encode(payload),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            
            if exchange_name:
                exchange = await channel.get_exchange(exchange_name)
                await exchange.publish(message, routing_key=routing_key)
            else:
                await channel.default_exchange.publish(message, routing_key=routing_key)

    async def publish_tick(self, data: dict):
        """Optimized market tick publication."""
        await self._publish(None, self.queue_name, data)

    async def publish_audit(self, payload: dict):
        """Secured audit log publication."""
        await self._publish(None, self.audit_queue, payload)

    async def publish_telemetry(self, payload: dict, routing_key: str = "telemetry.health"):
        """High-frequency telemetry publication."""
        await self._publish(self.telemetry_exchange, routing_key, payload)

    async def publish_signal(self, payload: dict):
        """ML Signal publication."""
        await self._publish(None, self.signal_topic, payload)

    async def publish_batch(self, batch: list[dict]):
        """Parallel batch publication via task gathering."""
        tasks = [self.publish_tick(data) for data in batch]
        await asyncio.gather(*tasks)

    async def consume_ticks(self, callback: Callable[[dict], Any], prefetch: int = 50):
        """
        High-throughput consumer loop with configurable prefetch.
        """
        if not self._channel_pool:
            await self.connect()

        async with self._channel_pool.acquire() as channel:
            await channel.set_qos(prefetch_count=prefetch)
            queue = await channel.get_queue(self.queue_name)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            data = msgspec.json.decode(message.body)
                            await callback(data)
                        except Exception as e:
                            logger.error("rabbitmq_consume_failed", error=str(e), queue=self.queue_name)
                            # Explicitly allow failure to bubble for message.process() to NACK
                            raise

    async def get_queue_stats(self, queue_name: str) -> dict[str, Any]:
        """Retrieve granular queue statistics using a pooled channel."""
        if not self._channel_pool:
            await self.connect()
        async with self._channel_pool.acquire() as channel:
            queue = await channel.declare_queue(queue_name, durable=True, passive=True)
            return {
                "message_count": queue.declaration_result.message_count,
                "consumer_count": queue.declaration_result.consumer_count,
            }

    async def close(self):
        """Graceful shutdown of all pools."""
        async with self._lock:
            if self._channel_pool:
                await self._channel_pool.close()
            if self._connection_pool:
                await self._connection_pool.close()
            self._connection_pool = None
            self._channel_pool = None
            logger.info("rabbitmq_pools_decommissioned")

    @property
    def connection(self):
        """Legacy compatibility property (Caution: returns None if using pools)."""
        return None # We use pools now

    @property
    def channel(self):
        """Legacy compatibility property."""
        return None # We use pools now


rabbitmq_manager = RabbitMQManager()


def get_rabbitmq():
    return rabbitmq_manager
