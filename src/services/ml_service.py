"""
ML Service (Singularity Refactored)
==================================

Interacts with the ML serving layer via gRPC or Shared Memory (SMI).
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


class MLService:
    """
    Handles ML inference via gRPC or high-speed Shared Memory.
    """

    def __init__(self):
        host, port = settings.ML_SERVICE_GRPC_URL.split(":")
        self.channel = Channel(host, int(port))
        self.stub = MLInferenceStub(self.channel)

        # 🚀 SINGULARITY: Initialize SHM for zero-copy inference
        self.shm = SHMManager("ml_inference_shm", dict, size=10 * 1024 * 1024)
        try:
            self.shm.create()
        except Exception:
            pass  # Already exists or managed by server

    async def predict(
        self, request: InferenceRequest, model_type: str = "xgb"
    ) -> InferenceResponse:
        start_time = time.perf_counter()

        # SOTA: Decide between gRPC and SMI (Shared Memory Inference)
        # For small requests, gRPC is fine. For large batches, SHM is god-mode.
        # Here we'll implement a hybrid: Write data to SHM, send name via gRPC.

        try:
            # 🚀 OPTIMIZATION: Write payload to SHM
            payload = request.model_dump()
            payload["model_type"] = model_type
            self.shm.write(payload)

            # Send 'SHM ticket' via gRPC (overloading implied_volatility field for POC)
            grpc_request = GrpcInferenceRequest(
                implied_volatility=-1.0,  # Flag for SHM-based inference
                model_type=self.shm.name,  # Passing SHM name here
            )

            # Execute gRPC call
            response = await self.stub.Predict(grpc_request, timeout=1.0)

            duration = (time.perf_counter() - start_time) * 1000
            ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)

            return InferenceResponse(
                price=response.price,
                model_type=response.model_type,
                latency_ms=duration,
            )

        except Exception as e:
            logger.error("ml_inference_failed", error=str(e))
            # Fallback to standard gRPC if SHM fails
            return await self._predict_grpc_fallback(request, model_type)

    async def _predict_grpc_fallback(
        self, request: InferenceRequest, model_type: str
    ) -> InferenceResponse:
        """Standard gRPC path without SHM."""
        grpc_request = GrpcInferenceRequest(
            underlying_price=request.underlying_price,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            is_call=request.is_call,
            model_type=model_type,
        )
        response = await self.stub.Predict(grpc_request, timeout=1.0)
        return InferenceResponse(
            price=response.price, model_type=response.model_type, latency_ms=0.0
        )

    async def close(self):
        self.channel.close()
        self.shm.close()


_ml_service = None


def get_ml_service() -> MLService:
    """Returns the singleton MLService instance, initializing it lazily."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service


class MLServiceProxy:
    """True lazy proxy that only instantiates the real service on first access."""

    def __getattr__(self, name):
        return getattr(get_ml_service(), name)


ml_service = MLServiceProxy()
