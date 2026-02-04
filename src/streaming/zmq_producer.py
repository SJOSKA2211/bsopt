import zmq
import zmq.asyncio
import orjson
import structlog
from typing import Dict, Any, Optional
from .base import Producer

logger = structlog.get_logger()

class ZMQMarketDataProducer(Producer):
    """
    Ultra-low latency ZeroMQ producer for internal high-speed data paths.
    Uses PUSH/PULL pattern for fire-and-forget data distribution.
    Complements Kafka by providing a sub-millisecond bypass for critical paths.
    """
    def __init__(self, endpoint: str = "tcp://*:5555"):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        
        # Performance tuning for high-frequency data
        self.socket.setsockopt(zmq.SNDHWM, 10000)  # High water mark
        self.socket.setsockopt(zmq.LINGER, 0)      # Don't wait on close
        
        try:
            self.socket.bind(endpoint)
            logger.info("zmq_producer_bound", endpoint=endpoint)
        except Exception as e:
            logger.error("zmq_bind_failed", endpoint=endpoint, error=str(e))
            raise

    async def produce(self, data: Dict[str, Any], **kwargs):
        """
        Send market data via ZeroMQ with zero-copy-like speed using orjson.
        """
        try:
            # key can be used for prefix-based filtering if switched to PUB/SUB
            # For PUSH/PULL we just send the payload
            payload = orjson.dumps(data)
            await self.socket.send(payload)
        except Exception as e:
            logger.error("zmq_send_error", error=str(e))

    def flush(self):
        """ZMQ is async/fire-and-forget, no explicit flush needed for PUSH."""
        pass

    def close(self):
        self.socket.close()
        self.context.term()
