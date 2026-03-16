"""
Transactional Outbox Service

Ensures reliable event delivery by polling the outbox table and 
dispatching events to Redis Streams or RabbitMQ.
"""

import asyncio
from datetime import UTC, datetime

import structlog
from sqlalchemy import select

from core.database import get_async_db_context
from core.database.models import OutboxEvent
from core.shared.redis_streams import RedisStreamManager

logger = structlog.get_logger(__name__)


class OutboxService:
    """
    Poller for the outbox table to dispatch events.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.stream_managers: dict[str, RedisStreamManager] = {}

    def _get_stream_manager(self, event_type: str) -> RedisStreamManager:
        # Route event types to specific streams
        stream_name = f"events_{event_type.split('.')[0]}"
        if stream_name not in self.stream_managers:
            self.stream_managers[stream_name] = RedisStreamManager(stream_name)
        return self.stream_managers[stream_name]

    async def dispatch_events(self) -> None:
        """Fetch pending events and publish them to Redis Streams."""
        async with get_async_db_context() as db:
            # 1. Fetch pending events
            result = await db.execute(
                select(OutboxEvent)
                .where(OutboxEvent.status == "pending")
                .order_by(OutboxEvent.created_at)
                .limit(50)
                .with_for_update(skip_locked=True)
            )
            events = result.scalars().all()

            if not events:
                return

            logger.info("dispatching_outbox_events", count=len(events))

            for event in events:
                try:
                    # 2. Publish to Redis Stream
                    manager = self._get_stream_manager(event.event_type)
                    data = {
                        "event_id": str(event.id),
                        "type": event.event_type,
                        "payload": event.payload,
                        "created_at": event.created_at.isoformat()
                    }
                    await manager.publish(data)

                    # 3. Mark as processed
                    event.status = "processed"
                    event.processed_at = datetime.now(UTC)
                except Exception as e:
                    logger.error("outbox_dispatch_failed", event_id=event.id, error=str(e))
                    event.status = "failed"

            await db.commit()

    async def run(self) -> None:
        """Main loop for the outbox dispatcher."""
        logger.info("starting_outbox_service")
        while True:
            try:
                await self.dispatch_events()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("outbox_service_loop_error", error=str(e))
                await asyncio.sleep(5)


outbox_service = OutboxService()
