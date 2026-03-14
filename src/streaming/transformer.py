import asyncio
import signal
import structlog
try:
    import bsopt_core
except ImportError:
    bsopt_core = None

from src.streaming.kafka_consumer import MarketDataConsumer
from src.streaming.kafka_producer import MarketDataProducer

logger = structlog.get_logger(__name__)

class MarketDataTransformer:
    """
    Kafka-to-Kafka Stream Processor for real-time feature engineering.
    Enriches raw market ticks with high-fidelity Greeks using bsopt_core.
    """

    def __init__(self):
        self.producer = MarketDataProducer()
        self.consumer = MarketDataConsumer(
            bootstrap_servers="kafka-1:9092",
            group_id="transformer-group",
            topics=["market-data"]
        )
        self.running = True

    async def transform_batch(self, batch: list[dict], topic: str):
        """
        Calculates Greeks for option ticks and enriches the stream.
        """
        transformed_batch = []
        for tick in batch:
            # 1. Check if it's an option that needs Greeks
            if tick.get("strike") and tick.get("expiry") and bsopt_core:
                try:
                    # Placeholder for actual spot retrieval or assumption
                    # In a real system, we'd fetch the underlying spot from Redis/SHM
                    spot = 100.0 # Mock spot for demonstration
                    
                    # Calculate Greeks via Rust
                    greeks = bsopt_core.black_scholes_greeks(
                        spot,
                        tick["strike"],
                        0.1, # Mock time to expiry
                        0.2, # Mock volatility
                        0.05, # Risk-free rate
                        0.0, # Dividend yield
                        tick["option_type"].lower() == "call"
                    )
                    
                    tick["delta"] = greeks.delta
                    tick["gamma"] = greeks.gamma
                    tick["vega"] = greeks.vega
                    tick["theta"] = greeks.theta
                except Exception as e:
                    logger.error("greeks_calculation_failed", symbol=tick.get("symbol"), error=str(e))

            transformed_batch.append(tick)

        if transformed_batch:
            await self.producer.produce_batch(transformed_batch, topic="transformed-market-data")
            logger.debug("batch_transformed_and_published", count=len(transformed_batch))

    async def run(self):
        logger.info("market_data_transformer_starting")
        try:
            await self.consumer.consume_messages(self.transform_batch, batch_size=100)
        except Exception as e:
            logger.error("transformer_runtime_error", error=str(e))
        finally:
            self.producer.close()
            logger.info("market_data_transformer_stopped")

    def stop(self):
        self.running = False
        self.consumer.stop()

async def main():
    transformer = MarketDataTransformer()

    loop = asyncio.get_running_loop()
    def shutdown():
        logger.info("shutdown_signal_received")
        transformer.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown)

    await transformer.run()

if __name__ == "__main__":
    asyncio.run(main())
