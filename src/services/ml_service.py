import structlog
import time
import random
import asyncio
from typing import Dict, Any, Optional, List
from src.config import settings
from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.api.exceptions import (
    ServiceUnavailableException,
    ValidationException,
    InternalServerException
)
from src.shared.observability import ML_PROXY_PREDICT_LATENCY
from src.utils.cache import multi_layer_cache
from src.utils.circuit_breaker import ml_client_circuit

# SOTA: Switch to grpclib for lower overhead
from grpclib.client import Channel
from src.protos.inference_grpc import MLInferenceStub
from src.protos.inference_pb2 import InferenceRequest as GrpcInferenceRequest

logger = structlog.get_logger(__name__)

class MLService:
    """
    Service for interacting with the Machine Learning serving layer.
    Exclusively uses grpclib for lightweight async communication.
    """
    def __init__(self):
        # Support comma-separated list of URLs for load balancing
        urls = getattr(settings, "ML_SERVICE_GRPC_URLS", settings.ML_SERVICE_GRPC_URL)
        self.grpc_urls = []
        if isinstance(urls, str):
            for u in urls.split(","):
                host, port = u.strip().split(":")
                self.grpc_urls.append((host, int(port)))
        else:
            host, port = urls.split(":")
            self.grpc_urls.append((host, int(port)))

        self._channels: List[Channel] = []
        self._stubs: List[MLInferenceStub] = []
        self._pool_size = getattr(settings, "ML_GRPC_POOL_SIZE", 4)
        self._lock = asyncio.Lock()

    async def _get_stub(self) -> MLInferenceStub:
        """Returns a stub from the pool using round-robin selection."""
        if not self._stubs:
            async with self._lock:
                if not self._stubs:
                    logger.info("initializing_grpclib_channel_pool", urls=self.grpc_urls, pool_size=self._pool_size)
                    for _ in range(self._pool_size):
                        host, port = random.choice(self.grpc_urls)
                        channel = Channel(host, port)
                        self._channels.append(channel)
                        self._stubs.append(MLInferenceStub(channel))
            
        return random.choice(self._stubs)

    @ml_client_circuit
    @multi_layer_cache(prefix="ml_predict", ttl=300, validation_model=InferenceResponse)
    async def predict(self, request: InferenceRequest, model_type: str = "xgb") -> InferenceResponse:
        """
        Proxies prediction request using an optimized grpclib stub.
        """
        start_time = time.perf_counter()
        
        try:
            stub = await self._get_stub()
            grpc_request = GrpcInferenceRequest(
                underlying_price=request.underlying_price,
                strike=request.strike,
                time_to_expiry=request.time_to_expiry,
                is_call=request.is_call,
                moneyness=request.moneyness,
                log_moneyness=request.log_moneyness,
                sqrt_time_to_expiry=request.sqrt_time_to_expiry,
                days_to_expiry=request.days_to_expiry,
                implied_volatility=request.implied_volatility,
                model_type=model_type
            )
            
            # grpclib call with timeout
            response = await stub.Predict(grpc_request, timeout=1.0)
            
            duration = (time.perf_counter() - start_time) * 1000
            ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)
            
            return InferenceResponse(
                price=response.price,
                model_type=response.model_type,
                latency_ms=float(response.latency_ms)
            )
            
        except Exception as e:
            logger.error("ml_inference_grpc_error", error=str(e))
            raise ServiceUnavailableException(message=f"ML Service Error: {str(e)}")

    async def close(self):
        """Gracefully close all channels."""
        for channel in self._channels:
            channel.close()
        self._channels = []
        self._stubs = []

# Singleton instance
ml_service = MLService()

def get_ml_service() -> MLService:
    return ml_service
