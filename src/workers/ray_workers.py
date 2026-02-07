import asyncio
import time

import ray
import structlog

from src.utils.distributed import RayOrchestrator
from src.utils.http_client import HttpClientManager

logger = structlog.get_logger(__name__)


@ray.remote
class MathActor:
    """
    SOTA: Persistent Ray Actor for high-speed mathematical computations.
    Maintains a persistent event loop and shared memory attachments.
    """

    def __init__(self):
        # 🚀 SINGULARITY: Initialize persistent state
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logger.info("math_actor_initialized")

    async def calibrate(self, symbol: str, data: list[dict]) -> dict:
        """🚀 SOTA: Non-blocking calibration logic."""
        start_time = time.time()
        # Simulated heavy math using the shared pricing kernels
        await asyncio.sleep(0.1)  # Mock work
        duration = (time.time() - start_time) * 1000
        logger.info("calibration_completed", symbol=symbol, latency_ms=duration)
        return {"status": "success", "symbol": symbol, "latency_ms": duration}

    def run_calibration(self, symbol: str, data: list[dict]):
        """Bridge sync Ray call to async actor logic."""
        return self.loop.run_until_complete(self.calibrate(symbol, data))


@ray.remote
class WebhookActor:
    """
    SOTA: Persistent Ray Actor for high-throughput webhook delivery.
    Leverages shared HttpClientManager with HTTP/2 and connection pooling.
    """

    def __init__(self):
        self.client = HttpClientManager.get_client()
        logger.info("webhook_actor_initialized")

    async def deliver(self, url: str, payload: dict):
        """🚀 SINGULARITY: Efficient delivery using pooled HTTP/2 connections."""
        try:
            response = await self.client.post(url, json=payload)
            logger.info("webhook_delivered", url=url, status=response.status_code)
            return response.status_code
        except Exception as e:
            logger.error("webhook_delivery_failed", url=url, error=str(e))
            return 500

    def run_delivery(self, url: str, payload: dict):
        return asyncio.run(self.deliver(url, payload))


if __name__ == "__main__":
    # Local test
    RayOrchestrator.init()
    actor = MathActor.remote()
    print(ray.get(actor.run_calibration.remote("SPY", [])))
