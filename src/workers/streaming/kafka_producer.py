"""
Market Data Kafka Producer

High-performance Kafka producer for market data streaming.
Routes validated ticks to primary topic and malformed ticks to DLQ.

Uses confluent-kafka for optimal performance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import structlog
from confluent_kafka import KafkaError, KafkaException, Producer

logger = structlog.get_logger(__name__)


@dataclass
class MarketTick:
    """Market tick data structure."""

    symbol: str
    price: float
    volume: int
    timestamp: float = field(default_factory=time.time)
    bid: Optional[float] = None
    ask: Optional[float] = None
    market: str = "unknown"
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "bid": self.bid,
            "ask": self.ask,
            "market": self.market,
            "source": self.source,
            "metadata": self.metadata,
        }

    def validate(self) -> tuple[bool, str | None]:
        """
        Validate tick data.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.symbol or len(self.symbol) > 20:
            return False, "Invalid symbol"

        if self.price <= 0:
            return False, "Price must be positive"

        if self.volume < 0:
            return False, "Volume cannot be negative"

        if self.timestamp <= 0 or self.timestamp > time.time() + 3600:
            return False, "Invalid timestamp"

        if self.bid is not None and self.bid <= 0:
            return False, "Bid must be positive"

        if self.ask is not None and self.ask <= 0:
            return False, "Ask must be positive"

        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            return False, "Bid cannot exceed ask"

        return True, None


@dataclass
class ProducerConfig:
    """Kafka producer configuration."""

    bootstrap_servers: str = "localhost:9092"
    client_id: str = "equaflow-producer"
    acks: str = "all"
    retries: int = 3
    retry_backoff_ms: int = 100
    linger_ms: int = 5
    batch_size: int = 16384
    compression_type: str = "lz4"
    enable_idempotence: bool = True
    max_in_flight_requests_per_connection: int = 5


class MarketDataProducer:
    """
    High-performance Kafka producer for market data.

    Features:
    - Automatic tick validation
    - DLQ routing for malformed ticks
    - Retry with exponential backoff
    - Idempotent delivery
    - Compression (LZ4)
    """

    def __init__(
        self,
        config: ProducerConfig | None = None,
        dlq_topic: str = "market-ticks-dlq",
    ):
        self.config = config or ProducerConfig()
        self.dlq_topic = dlq_topic
        self._producer: Producer | None = None
        self._delivery_reports: list[dict[str, Any]] = []
        self._stats_callback_enabled = False

    def _get_producer_config(self) -> dict[str, Any]:
        """Build producer configuration."""
        return {
            "bootstrap.servers": self.config.bootstrap_servers,
            "client.id": self.config.client_id,
            "acks": self.config.acks,
            "retries": self.config.retries,
            "retry.backoff.ms": self.config.retry_backoff_ms,
            "linger.ms": self.config.linger_ms,
            "batch.size": self.config.batch_size,
            "compression.type": self.config.compression_type,
            "enable.idempotence": self.config.enable_idempotence,
            "max.in.flight.requests.per.connection": self.config.max_in_flight_requests_per_connection,
            "on_delivery": self._delivery_callback,
        }

    def _delivery_callback(self, err: KafkaError | None, msg) -> None:
        """Callback for message delivery reports."""
        report = {
            "timestamp": time.time(),
            "error": str(err) if err else None,
            "topic": msg.topic() if msg else None,
            "partition": msg.partition() if msg else None,
            "offset": msg.offset() if msg else None,
        }
        self._delivery_reports.append(report)

        if len(self._delivery_reports) > 1000:
            self._delivery_reports = self._delivery_reports[-500:]

    def start(self) -> None:
        """Initialize the Kafka producer."""
        if self._producer is not None:
            logger.warning("producer_already_started")
            return

        try:
            self._producer = Producer(self._get_producer_config())
            logger.info(
                "kafka_producer_started",
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
            )
        except KafkaException as e:
            logger.error("kafka_producer_init_failed", error=str(e))
            raise

    def stop(self, timeout: float = 10.0) -> None:
        """Flush and close the producer."""
        if self._producer is None:
            return

        try:
            remaining = self._producer.flush(timeout)
            if remaining > 0:
                logger.warning(
                    "kafka_flush_incomplete",
                    remaining_messages=remaining,
                )
            logger.info("kafka_producer_stopped")
        except Exception as e:
            logger.error("kafka_producer_stop_error", error=str(e))
        finally:
            self._producer = None

    def _send_to_dlq(self, tick: MarketTick, error: str) -> None:
        """Route malformed tick to Dead Letter Queue."""
        if self._producer is None:
            logger.error("producer_not_started_cannot_send_dlq")
            return

        dlq_message = {
            "original_tick": tick.to_dict(),
            "error": error,
            "error_timestamp": time.time(),
            "dlq_topic": self.dlq_topic,
            "trace_id": str(uuid4()),
        }

        try:
            self._producer.produce(
                self.dlq_topic,
                key=tick.symbol.encode("utf-8"),
                value=json.dumps(dlq_message).encode("utf-8"),
                callback=self._delivery_callback,
            )
            self._producer.poll(0)
            logger.warning(
                "tick_routed_to_dlq",
                symbol=tick.symbol,
                error=error,
            )
        except BufferError:
            logger.warning("producer_buffer_full_waiting")
            self._producer.poll(1)
            self._producer.produce(
                self.dlq_topic,
                key=tick.symbol.encode("utf-8"),
                value=json.dumps(dlq_message).encode("utf-8"),
            )
        except Exception as e:
            logger.error("dlq_send_failed", error=str(e), symbol=tick.symbol)

    def send_tick(
        self,
        tick: MarketTick,
        topic: str = "market-ticks",
        sync: bool = False,
    ) -> bool:
        """
        Send validated tick to Kafka topic.

        Args:
            tick: MarketTick data
            topic: Target Kafka topic
            sync: Wait for delivery confirmation

        Returns:
            True if sent successfully, False otherwise
        """
        if self._producer is None:
            self.start()

        is_valid, error = tick.validate()
        if not is_valid:
            self._send_to_dlq(tick, error or "Validation failed")
            return False

        try:
            self._producer.produce(
                topic,
                key=tick.symbol.encode("utf-8"),
                value=json.dumps(tick.to_dict()).encode("utf-8"),
                callback=self._delivery_callback,
            )

            if sync:
                self._producer.poll(timeout=10.0)

            self._producer.poll(0)
            return True

        except BufferError:
            logger.warning("producer_buffer_full_waiting")
            while True:
                self._producer.poll(1)
                try:
                    self._producer.produce(
                        topic,
                        key=tick.symbol.encode("utf-8"),
                        value=json.dumps(tick.to_dict()).encode("utf-8"),
                    )
                    return True
                except BufferError:
                    continue

        except KafkaException as e:
            logger.error(
                "kafka_send_failed",
                error=str(e),
                symbol=tick.symbol,
            )
            self._send_to_dlq(tick, f"Kafka error: {e}")
            return False

    def send_batch(
        self,
        ticks: list[MarketTick],
        topic: str = "market-ticks",
    ) -> tuple[int, int]:
        """
        Send batch of ticks to Kafka.

        Args:
            ticks: List of MarketTick data
            topic: Target Kafka topic

        Returns:
            Tuple of (success_count, failure_count)
        """
        success = 0
        failure = 0

        for tick in ticks:
            if self.send_tick(tick, topic):
                success += 1
            else:
                failure += 1

        return success, failure

    def get_stats(self) -> dict[str, Any]:
        """Get producer statistics."""
        if self._producer is None:
            return {"status": "not_started"}

        return {
            "status": "running",
            "delivery_reports_queued": len(self._delivery_reports),
            "out_queue_length": self._producer._queue_buffer.qsize()
            if hasattr(self._producer, "_queue_buffer")
            else 0,
        }


class MultiTopicProducer:
    """
    Producer that routes ticks to different topics based on symbol patterns.
    """

    def __init__(self, config: ProducerConfig | None = None):
        self.producer = MarketDataProducer(config)

        self.topic_routes: dict[str, str] = {
            "^AAPL$": "market-ticks-equity",
            "^GOOGL$": "market-ticks-equity",
            "^MSFT$": "market-ticks-equity",
            ".*-OPTIONS$": "market-ticks-options",
            ".*-FUTURES$": "market-ticks-futures",
        }
        self.default_topic = "market-ticks"

    def start(self) -> None:
        """Start the producer."""
        self.producer.start()

    def stop(self) -> None:
        """Stop the producer."""
        self.producer.stop()

    def _get_topic(self, symbol: str) -> str:
        """Determine topic based on symbol pattern."""
        import re

        for pattern, topic in self.topic_routes.items():
            if re.match(pattern, symbol):
                return topic
        return self.default_topic

    def send_tick(self, tick: MarketTick) -> bool:
        """Route tick to appropriate topic based on symbol."""
        topic = self._get_topic(tick.symbol)
        return self.producer.send_tick(tick, topic)


if __name__ == "__main__":
    import random

    config = ProducerConfig(
        bootstrap_servers="localhost:9092",
        client_id="test-producer",
    )

    producer = MarketDataProducer(config)

    try:
        producer.start()
        print("Producer started, sending test ticks...")

        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        for i in range(100):
            tick = MarketTick(
                symbol=random.choice(symbols),
                price=100 + random.uniform(-5, 5),
                volume=random.randint(100, 10000),
                market="NASDAQ",
                source="test",
            )

            result = producer.send_tick(tick)
            print(f"Sent tick {i + 1}: {tick.symbol} @ {tick.price:.2f} -> {result}")

            if i % 10 == 0:
                producer._producer.poll(1)

        print("\nFlushing...")
        producer.stop()
        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
