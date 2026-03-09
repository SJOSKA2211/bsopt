import asyncio
from typing import Any

import orjson
import structlog
from confluent_kafka import Consumer, Producer

from src.ml.reinforcement_learning.augmented_agent import SentimentExtractor

logger = structlog.get_logger(__name__)


class SentimentIngestor:
    """
    Ingests news/social media data, extracts sentiment, and publishes signals.
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "scraper.news"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_group = "sentiment-ingestor"

        # Initialize ML model
        self.extractor = SentimentExtractor()

        # Initialize Kafka clients
        # Note: In tests these are patched
        self.producer = Producer({"bootstrap.servers": self.bootstrap_servers})
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": self.consumer_group,
                "auto.offset.reset": "earliest",
            }
        )
        self.consumer.subscribe([self.topic])

    async def process_batch(self, messages: list[bytes]) -> None:
        """
        God-Mode: Batch process messages for high throughput.
        """
        results = []
        for msg in messages:
            try:
                data = orjson.loads(msg)
                text = data.get("text", "")
                if text:
                    score = self.extractor.get_sentiment_score(text)
                    results.append(
                        {
                            "symbol": data.get("symbol", "GLOBAL"),
                            "sentiment": score,
                            "timestamp": data.get("timestamp"),
                        }
                    )
            except Exception:
                continue

        if results and self.producer:
            # High-speed batch publishing
            for res in results:
                self.producer.produce(
                    "model.signals",
                    key=res["symbol"].encode("utf-8"),
                    value=orjson.dumps(res),
                )
            self.producer.flush()
            logger.info("sentiment_batch_processed", count=len(results))

    def run(self, batch_size: int = 10):
        """
        Main high-performance consumption loop with batching.
        """
        logger.info("sentiment_ingestor_loop_start", batch_size=batch_size)
        messages = []
        try:
            while True:
                msg = self.consumer.poll(0.1)
                if msg is not None:
                    if not msg.error():
                        messages.append(msg.value())

                if len(messages) >= batch_size or (messages and msg is None):
                    asyncio.run(self.process_batch(messages))
                    messages = []

        except Exception as e:
            logger.error("sentiment_ingestor_crashed", error=str(e))
            raise
        finally:
            self.consumer.close()


class SentimentPipeline:
    """
    Data Pipeline connecting Scraper Service outputs to Sentiment Oracle.
    Processes unstructured text into actionable signals for the RL Agent.
    """

    def __init__(self):
        self.extractor = SentimentExtractor()
        logger.info("sentiment_pipeline_initialized")

    async def process_scraper_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Processes a single message from the scraper.

        Args:
            message (Dict[str, Any]): Data containing 'text', 'symbol', etc.

        Returns:
            Dict[str, Any]: Augmented message with 'sentiment' score.
        """
        text = message.get("text", "")
        symbol = message.get("symbol", "GLOBAL")

        if not text:
            return {**message, "sentiment": 0.0}

        try:
            # OPTIMIZED: Offload blocking NLP extraction to a thread pool
            sentiment_score = await asyncio.to_thread(self.extractor.get_sentiment_score, text)

            logger.info("sentiment_extracted", symbol=symbol, score=sentiment_score)

            return {**message, "sentiment": sentiment_score}
        except Exception as e:
            logger.error(f"sentiment_extraction_failed: {e}")
            return {**message, "sentiment": 0.0}

    async def run_consumer(self):
        """
        Runs the Kafka consumer loop for real-time sentiment extraction.
        """
        logger.info("sentiment_pipeline_starting_consumer")
        # Initialize and run the ingestor
        ingestor = SentimentIngestor()
        # Note: SentimentIngestor.run() is synchronous/blocking,
        # so we run it in a thread to not block the event loop if called from async code
        await asyncio.to_thread(ingestor.run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Sentiment Ingestor")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--topic", type=str, default="scraper.news")
    parser.add_argument("--broker", type=str, default="kafka-1:9092")

    args = parser.parse_args()

    ingestor = SentimentIngestor(bootstrap_servers=args.broker, topic=args.topic)
    ingestor.run(batch_size=args.batch_size)
