import json
import structlog
from aiopika import connect_robust, Message

logger = structlog.get_logger(__name__)


class DLQManager:
    """
    Manages Dead Letter Queues for malformed or failed data ingestion.
    Uses RabbitMQ as the DLQ sink.
    """

    def __init__(self):
        self.amqp_url = "amqp://bsopt_admin:bsopt_rmq_secret@rabbitmq:5672/"
        self.connection = None
        self.channel = None

    async def _ensure_connection(self):
        if not self.connection or self.connection.is_closed:
            self.connection = await connect_robust(self.amqp_url)
            self.channel = await self.connection.channel()
            # Declare DLQ exchange and queue
            await self.channel.declare_exchange("dlq_exchange", type="direct", durable=True)
            await self.channel.declare_queue("ingestion_dlq", durable=True)
            await self.channel.queue_bind(
                "ingestion_dlq", "dlq_exchange", routing_key="malformed_ticks"
            )

    async def send_to_dlq(self, payload: dict, reason: str, original_source: str):
        """
        Pushes a failed record to the DLQ with metadata.
        """
        try:
            await self._ensure_connection()

            enrichment = {
                "error_reason": reason,
                "original_source": original_source,
                "ingestion_timestamp": str(structlog.get_logger()._get_timestamp()),
                "raw_data": payload,
            }

            await self.channel.default_exchange.publish(
                Message(
                    body=json.dumps(enrichment).encode(),
                    delivery_mode=2,  # Persistent
                ),
                routing_key="ingestion_dlq",
            )

            logger.warning("pushed_to_dlq", symbol=payload.get("symbol"), reason=reason)

        except Exception as e:
            logger.error("dlq_publish_critical_failure", error=str(e), payload=payload)


dlq_manager = DLQManager()
