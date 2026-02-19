import time

import numpy as np
import ray
import structlog

from src.utils.distributed import RayOrchestrator
from src.utils.http_client import HttpClientManager

logger = structlog.get_logger(__name__)


@ray.remote
class MathActor:
    """
    OPTIMIZED: Async-native Ray Actor for high-speed batch mathematics.
    Uses vectorized silicon kernels for O(1) dispatched computation.
    """
    def __init__(self):
        from src.pricing.factory import PricingEngineFactory
        self.engine = PricingEngineFactory.get_engine("black_scholes")
        logger.info("math_actor_ready", engine="vectorized_bs")

    async def calibrate_batch(self, spots: np.ndarray, strikes: np.ndarray, times: np.ndarray, vols: np.ndarray, rates: np.ndarray) -> np.ndarray:
        """Vectorized calibration dispatched to machine-code kernels."""
        start_time = time.perf_counter()
        
        # Dispatch to vectorized JIT kernel (O(1) from Python's perspective)
        prices = self.engine.price_options(spots, strikes, times, vols, rates, 0.0, "call")
        
        duration = (time.perf_counter() - start_time) * 1000
        logger.debug("batch_calibration_complete", size=len(spots), ms=round(duration, 3))
        return prices

@ray.remote
class WebhookActor:
    """
    OPTIMIZED: Async-native Webhook Actor with concurrency control.
    """
    def __init__(self):
        self.client = HttpClientManager.get_client()
        self.semaphore = HttpClientManager.get_semaphore(limit=50)
        logger.info("webhook_actor_ready")

    async def deliver(self, url: str, payload: dict) -> int:
        """Asynchronous delivery with backpressure awareness."""
        async with self.semaphore:
            try:
                response = await self.client.post(url, json=payload, timeout=5.0)
                return response.status_code
            except Exception as e:
                logger.error("webhook_delivery_failed", url=url, error=str(e))
                return 500



if __name__ == "__main__":
    # Local test
    RayOrchestrator.init()
    actor = MathActor.remote()
    print(ray.get(actor.run_calibration.remote("SPY", [])))
