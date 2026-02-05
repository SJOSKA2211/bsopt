import asyncio
from typing import Any

import orjson
import structlog
from confluent_kafka import Consumer, KafkaError, Producer

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
        self.producer = Producer({'bootstrap.servers': self.bootstrap_servers})
        self.consumer = Consumer({
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.consumer_group,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([self.topic])

    async def process_news_message(self, message_bytes: bytes) -> None:
        """
        Process a single message payload using optimized serialization.
        """
        try:
            data = orjson.loads(message_bytes)
        except Exception:
            logger.error("failed_to_decode_message")
            return

        text = data.get("text", "")
        if not text:
            return

        # Extract sentiment
        score = self.extractor.get_sentiment_score(text)
        
        # Publish result
        result = {
            "original_data": data,
            "sentiment_score": score,
            "signal_type": "sentiment"
        }
        
        # Check if producer is available (might be mocked)
        if self.producer:
            self.producer.produce(
                "model.signals",
                key=data.get("symbol", "GLOBAL").encode('utf-8'),
                value=orjson.dumps(result)
            )
            self.producer.poll(0)

    def run(self):
        """
        Main high-performance consumption loop.
        """
        logger.info("sentiment_ingestor_loop_start")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    error = msg.error()
                    if hasattr(error, 'code') and error.code() == KafkaError._PARTITION_EOF:
                        continue
                    
                    logger.error("kafka_consumer_error", error=str(error))
                    continue
                
                # Process message
                asyncio.run(self.process_news_message(msg.value()))

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
            # Perform extraction (blocking call in this stub, should be optimized)
            sentiment_score = self.extractor.get_sentiment_score(text)
            
            logger.info("sentiment_extracted", symbol=symbol, score=sentiment_score)
            
            return {
                **message,
                "sentiment": sentiment_score
            }
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