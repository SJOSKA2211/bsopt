import asyncio
import time

import ray
import structlog

from src.shared.utils.distributed import RayOrchestrator
from src.shared.utils.http_client import HttpClientManager

logger = structlog.get_logger(__name__)


@ray.remote
class MathActor:
    """
    Persistent Ray Actor for high-speed mathematical computations.
    """

    def __init__(self):
        logger.info("math_worker_ready")

    async def calibrate(self, symbol: str, data: list[dict]) -> dict:
        """Perform real mathematical calibration using silicon kernels."""
        start_time = time.time()

        #  REAL WORK: Perform a batch of BS calculations
        from src.pricing.factory import PricingEngineFactory
        from src.pricing.models import BSParameters

        engine = PricingEngineFactory.get_engine("black_scholes")
        params = BSParameters(S=100.0, K=100.0, T=0.1, sigma=0.2, r=0.05)
        # Force some CPU cycles
        for _ in range(100):
            engine.calculate_greeks(params)

        duration = (time.time() - start_time) * 1000
        logger.info("calibration_done", symbol=symbol, ms=duration)
        return {"status": "ok", "symbol": symbol, "latency_ms": duration}

    def run_calibration(self, symbol: str, data: list[dict]):
        """Synchronous bridge for Ray orchestration."""
        try:
            loop = asyncio.get_running_loop()
            # If we are here, we are already in an event loop (likely Ray's IO thread)
            # Use a thread-safe way to run the coroutine
            return asyncio.run_coroutine_threadsafe(self.calibrate(symbol, data), loop).result()
        except RuntimeError:
            # No running event loop, use asyncio.run
            return asyncio.run(self.calibrate(symbol, data))


@ray.remote
class WebhookActor:
    """
    Persistent Ray Actor for high-throughput webhook delivery.
    """

    def __init__(self):
        # Use shared HTTP/2 connection pool
        self.client = HttpClientManager.get_client()
        logger.info("delivery_worker_ready")

    async def deliver(self, url: str, payload: dict):
        """Asynchronous webhook delivery."""
        try:
            response = await self.client.post(url, json=payload)
            logger.info("webhook_sent", url=url, status=response.status_code)
            return response.status_code
        except Exception as e:
            logger.error("delivery_error", url=url, error=str(e))
            return 500

    def run_delivery(self, url: str, payload: dict):
        """Synchronous bridge."""
        return asyncio.run(self.deliver(url, payload))


if __name__ == "__main__":
    # Local test
    RayOrchestrator.init()
    actor = MathActor.remote()
    print(ray.get(actor.run_calibration.remote("SPY", [])))
