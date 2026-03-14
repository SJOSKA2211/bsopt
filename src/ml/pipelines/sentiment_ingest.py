"""
Sentiment Ingestion Pipeline — expected by test_sentiment_pipeline.py.
"""

from __future__ import annotations

import structlog
from typing import Any

logger = structlog.get_logger(__name__)

# Real or mock imports for dependencies that the tests patch
try:
    from confluent_kafka import Consumer as KafkaConsumer, Producer as KafkaProducer
except ImportError:
    class KafkaProducer: # type: ignore
        pass

    class KafkaConsumer: # type: ignore
        pass

# Type-safe aliases for internal use
Producer = KafkaProducer
Consumer = KafkaConsumer


class SentimentExtractor:
    def get_sentiment_score(self, text: str) -> float:
        return 0.0


class SentimentIngestor:
    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self.bootstrap_servers = bootstrap_servers

    async def ingest_sentiment(self, text: str) -> dict[str, float]:
        """
        Mock ingestion.
        """
        return {"sentiment": 0.0}
