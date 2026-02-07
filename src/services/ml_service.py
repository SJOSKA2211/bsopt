"""
Machine Learning Service
========================

Handles ML inference via gRPC or high-speed Shared Memory (SMI).
"""

import time

import structlog
from grpclib.client import Channel

from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.config import settings
from src.protos.inference_grpc import MLInferenceStub
from src.protos.inference_pb2 import InferenceRequest as GrpcInferenceRequest
from src.shared.observability import ML_PROXY_PREDICT_LATENCY
from src.shared.shm_manager import SHMManager

logger = structlog.get_logger(__name__)

# Constants for hybrid inference protocol
SHM_INFERENCE_FLAG = -1.0

class MLService:
    """
    Unified ML inference interface.
    """
    def __init__(self):
        host, port = settings.ML_SERVICE_GRPC_URL.split(":")
        self.channel = Channel(host, int(port))
        self.stub = MLInferenceStub(self.channel)
        
        # Initialize Shared Memory for zero-copy bulk inference
        self.shm = SHMManager("ml_inference_shm", dict, size=10 * 1024 * 1024)
        try:
            self.shm.create()
        except Exception:
            pass # Already managed by inference server

    async def predict(self, request: InferenceRequest, model_type: str = "xgb") -> InferenceResponse:
        start_time = time.perf_counter()
        
        try:
            # Hybrid SMI Path: Write payload to SHM, notify via gRPC
            payload = request.model_dump()
            payload["model_type"] = model_type
            self.shm.write(payload)
            
            grpc_request = GrpcInferenceRequest(
                implied_volatility=SHM_INFERENCE_FLAG,
                model_type=self.shm.name
            )
            
            response = await self.stub.Predict(grpc_request, timeout=1.0)
            
            duration = (time.perf_counter() - start_time) * 1000
            ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)
            
            return InferenceResponse(
                price=response.price,
                model_type=response.model_type,
                latency_ms=duration
            )
            
        except Exception as e:
            logger.warning("shm_inference_failed_falling_back", error=str(e))
            return await self._predict_grpc_fallback(request, model_type)

    async def _predict_grpc_fallback(self, request: InferenceRequest, model_type: str) -> InferenceResponse:
        """Standard gRPC path."""
        grpc_request = GrpcInferenceRequest(
            underlying_price=request.underlying_price,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            is_call=request.is_call,
            model_type=model_type
        )
        response = await self.stub.Predict(grpc_request, timeout=1.0)
        return InferenceResponse(price=response.price, model_type=response.model_type, latency_ms=0.0)

    async def close(self):
        self.channel.close()
        self.shm.close()

_ml_service_instance: MLService | None = None

def get_ml_service() -> MLService:
    """Returns the singleton MLService instance."""
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance

class MLServiceProxy:
    """Lazy proxy for service injection."""
    def __getattr__(self, name):
        return getattr(get_ml_service(), name)

ml_service = MLServiceProxy()