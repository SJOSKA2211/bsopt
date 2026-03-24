import asyncio
import json
import logging
from typing import Any, Callable

import aio_pika
from tenacity import retry, stop_after_attempt, wait_exponential

from src.shared.config import settings

logger = logging.getLogger(__name__)

class RabbitMQManager:
    """
    Production-Grade RabbitMQ Manager with DLQ and Retry logic.
    """

    def __init__(self):
        self.url = settings.RABBITMQ_URL
        self.connection = None
        self.channel = None
        self.exchange_name = "market_data"
        self.queue_name = "market_ticks"
        self.dlq_name = "market_ticks_dlq"
        self.audit_exchange = "audit_exchange"
        self.audit_queue = "audit_logs"
        self.news_topic = "scraper.news"
        self.signal_topic = "model.signals"

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def connect(self):
        """Establish connection with exponential backoff."""
        if not self.connection or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            
            # Declare DLQ
            await self.channel.declare_queue(self.dlq_name, durable=True)
            
            # Declare main queue with DLQ routing
            await self.channel.declare_queue(
                self.queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": self.dlq_name,
                }
            )
            
            # Declare Audit Queue
            await self.channel.declare_queue(self.audit_queue, durable=True)
            await self.channel.declare_exchange(self.audit_exchange, type="direct", durable=True)
            await self.channel.bind_queue(self.audit_queue, self.audit_exchange, routing_key="audit")
            
            # Declare Sentiment/News Queues
            await self.channel.declare_queue(self.news_topic, durable=True)
            await self.channel.declare_queue(self.signal_topic, durable=True)
            
            logger.info("rabbitmq_connected_and_queues_declared")

    async def publish_tick(self, data: dict):
        """Publish a market tick to the queue."""
        if not self.channel:
            await self.connect()
            
        message = aio_pika.Message(
            body=json.dumps(data, default=str).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message, routing_key=self.queue_name
        )

    async def publish_audit(self, payload: dict):
        """Publish an audit log to the audit queue."""
        if not self.channel:
            await self.connect()
            
        message = aio_pika.Message(
            body=json.dumps(payload, default=str).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message, routing_key=self.audit_queue
        )

    async def publish_signal(self, payload: dict):
        """Publish a sentiment signal to the signals queue."""
        if not self.channel:
            await self.connect()
            
        message = aio_pika.Message(
            body=json.dumps(payload, default=str).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.channel.default_exchange.publish(
            message, routing_key=self.signal_topic
        )

    async def publish_batch(self, batch: list[dict]):
        """Publish a batch of market ticks to the queue."""
        if not self.channel:
            await self.connect()
            
        # Use a transaction-like approach or just gather the publishes
        # For aio-pika, simple gather is often sufficient for performance
        tasks = [self.publish_tick(data) for data in batch]
        await asyncio.gather(*tasks)

    async def consume_ticks(self, callback: Callable[[dict], Any]):
        """Consume ticks from the queue and trigger callback."""
        if not self.channel:
            await self.connect()
            
        queue = await self.channel.get_queue(self.queue_name)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        await callback(data)
                    except Exception as e:
                        logger.error("rabbitmq_consume_failed_routing_to_dlq", error=str(e))
                        # message.process() automatically nacks/rejects on exception 
                        # if not handled, but here we want explicit control if needed.
                        raise e

    async def close(self):
        """Cleanly close connection."""
        if self.connection:
            await self.connection.close()
            logger.info("rabbitmq_connection_closed")

rabbitmq_manager = RabbitMQManager()

def get_rabbitmq():
    return rabbitmq_manager
