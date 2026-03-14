import json

import aio_pika
import aiokafka
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class DeadLetterQueue:
    def __init__(self, broker_type: str = "kafka"):
        self.broker_type = broker_type
        self.kafka_producer = None
        self.rabbitmq_connection = None

    async def initialize(self):
        if self.broker_type == "kafka":
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS
            )
            await self.kafka_producer.start()
        elif self.broker_type == "rabbitmq":
            self.rabbitmq_connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)

    async def send_to_dlq(self, message: dict, reason: str, original_topic: str):
        logger.warning("sending_message_to_dlq", reason=reason, topic=original_topic)
        payload = {
            "original_message": message,
            "error_reason": reason,
            "original_topic": original_topic,
            "timestamp": json.dumps(settings.current_timestamp()),  # Utility if exists
        }

        if self.broker_type == "kafka" and self.kafka_producer:
            await self.kafka_producer.send_and_wait(
                "dlq-topic", json.dumps(payload).encode("utf-8")
            )
        elif self.broker_type == "rabbitmq" and self.rabbitmq_connection:
            async with self.rabbitmq_connection.channel() as channel:
                await channel.default_exchange.publish(
                    aio_pika.Message(body=json.dumps(payload).encode()), routing_key="dlq-queue"
                )

    async def close(self):
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()


dlq_manager = DeadLetterQueue(broker_type="kafka")
