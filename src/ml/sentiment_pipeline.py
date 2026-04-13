import asyncio

import msgspec
import structlog

from src.ml.reinforcement_learning.augmented_agent import SentimentExtractor
from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)


class SentimentIngestor:
    """
    Ingests news/social media data, extracts sentiment, and publishes signals.
    OPTIMIZED: Uses RabbitMQ for high-performance async messaging.
    """

    def __init__(self, topic: str = "scraper.news"):
        self.topic = topic
        self.rmq = get_rabbitmq()
        # Initialize ML model
        self.extractor = SentimentExtractor()

    async def process_batch(self, messages: list[dict]) -> None:
        """
        High-Performance: Batch process messages for high throughput.
        """
        results = []
        for data in messages:
            try:
                text = data.get("text", "")
                if text:
                    # Offload CPU-bound NLP to thread pool
                    score = await asyncio.to_thread(self.extractor.get_sentiment_score, text)
                    results.append(
                        {
                            "symbol": data.get("symbol", "GLOBAL"),
                            "sentiment": score,
                            "timestamp": data.get("timestamp"),
                        }
                    )
            except Exception:
                continue

        if results:
            # High-speed parallel publishing to RabbitMQ
            tasks = [self.rmq.publish_signal(res) for res in results]
            await asyncio.gather(*tasks)
            logger.info("sentiment_batch_processed", count=len(results))

    async def run(self, batch_size: int = 10):
        """
        Main high-performance async consumption loop from RabbitMQ.
        """
        logger.info("sentiment_ingestor_loop_start", batch_size=batch_size)

        async def callback(data: dict):
            # For RabbitMQ, we handle messages via callback or iterator
            # Here we'll wrap it to accumulate batches if needed, or just process immediately
            await self.process_batch([data])

        # Consume from the news topic
        # Note: We need a specialized consumer for news if it's not the default tick stream
        # RabbitMQManager.consume_ticks uses 'market_ticks' queue.
        # We'll use a local consumer here for the news topic.

        if not self.rmq.channel:
            await self.rmq.connect()

        queue = await self.rmq.channel.get_queue(self.topic)
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = msgspec.json.decode(message.body)
                        await self.process_batch([data])
                    except Exception as e:
                        logger.error("news_consume_failed", error=str(e))


class SentimentPipeline:
    """
    Data Pipeline connecting Scraper Service outputs to Sentiment Oracle.
    """

    def __init__(self):
        self.extractor = SentimentExtractor()
        logger.info("sentiment_pipeline_initialized")

    async def run_consumer(self):
        """
        Runs the RabbitMQ consumer loop for real-time sentiment extraction.
        """
        logger.info("sentiment_pipeline_starting_consumer")
        ingestor = SentimentIngestor()
        await ingestor.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Sentiment Ingestor")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--topic", type=str, default="scraper.news")

    args = parser.parse_args()

    async def main():
        ingestor = SentimentIngestor(topic=args.topic)
        await ingestor.run(batch_size=args.batch_size)

    asyncio.run(main())