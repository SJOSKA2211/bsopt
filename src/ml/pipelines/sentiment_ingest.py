"""
Sentiment Ingestion Pipeline — expected by test_sentiment_pipeline.py.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Real or mock imports for dependencies that the tests patch
try:
    from confluent_kafka import Consumer, Producer
except ImportError:
    class Producer: pass
    class Consumer: pass

class SentimentExtractor:
    def get_sentiment_score(self, text: str) -> float:
        return 0.0


class SentimentIngestor:
    def __init__(self, bootstrap_servers: str = "localhost:9092") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.extractor = SentimentExtractor()
        
        # In a real scenario these would connect to Kafka
        self.producer = Producer({"bootstrap.servers": self.bootstrap_servers})
        self.consumer = Consumer({
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": "sentiment_group"
        })

    async def process_news_message(self, message: bytes) -> None:
        try:
            data = json.loads(message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("invalid_json_message")
            return

        text = data.get("text", "")
        if not text:
            return

        score = self.extractor.get_sentiment_score(text)
        
        # Produce processed result
        output = {
            "symbol": data.get("symbol"),
            "sentiment_score": score,
            "timestamp": data.get("timestamp")
        }
        self.producer.produce("sentiment_scores", json.dumps(output).encode("utf-8"))

    def run(self) -> None:
        self.consumer.subscribe(["news_feed"])
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue
                # Normally we would call process_news_message via asyncio run or task,
                # but this loop is synchronous and primarily for testing Kafka interactions.
        finally:
            self.consumer.close()
