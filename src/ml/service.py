"""
Machine Learning Service
Enhanced with High-Performance Persistence and Vectorized Database Ingestion.
"""

import asyncio
import time
from math import erf

import grpc
import structlog

from api.schemas.ml import InferenceRequest, InferenceResponse
from src.database.pipeliner import db_engine
from src.shared.config import settings
from src.shared.observability import ML_PROXY_PREDICT_LATENCY
from src.shared.protos import inference_pb2, inference_pb2_grpc

logger = structlog.get_logger(__name__)


class MLService:
    """
    Enhanced ML pricing service with automated hypertable persistence.
    OPTIMIZED: Persistent gRPC connection to the ML Inference Manifold.
    """

    def __init__(self):
        self.grpc_url = settings.ML_SERVICE_GRPC_URL
        self._channel = None
        self._stub = None
        logger.info("ml_service_initialized", grpc_url=self.grpc_url)

    async def _get_grpc_stub(self):
        """Lazy initialization of persistent gRPC stub."""
        if self._channel is None:
            self._channel = grpc.aio.insecure_channel(
                self.grpc_url,
                options=[
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )
            self._stub = inference_pb2_grpc.MLInferenceStub(self._channel)
        return self._stub

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using math.erf (Optimized)."""
        return 0.5 * (1.0 + erf(x * 0.7071067811865476))  # 1/sqrt(2)

    def _black_scholes_price(self, req: InferenceRequest) -> float:
        # Optimized Black-Scholes using src.shared engines
        from src.math_kernel.black_scholes import BlackScholesEngine

        try:
            price = BlackScholesEngine.price_options(
                spot=req.underlying_price,
                strike=req.strike,
                maturity=req.time_to_expiry,
                volatility=req.implied_volatility or 0.25,
                rate=0.01,
                option_type="call" if req.is_call else "put",
            )
            return float(price)
        except Exception:
            # Absolute fallback
            intrinsic = (
                max(req.underlying_price - req.strike, 0.0)
                if req.is_call
                else max(req.strike - req.underlying_price, 0.0)
            )
            return intrinsic * 0.9

    async def predict(
        self, request: InferenceRequest, model_type: str = "xgb", symbol: str = "UNKNOWN"
    ) -> InferenceResponse:
        start_time = time.perf_counter()

        # 1. Attempt gRPC Inference (Primary)
        try:
            stub = await self._get_grpc_stub()
            grpc_req = inference_pb2.InferenceRequest(
                underlying_price=request.underlying_price,
                strike=request.strike,
                time_to_expiry=request.time_to_expiry,
                is_call=bool(request.is_call),
                moneyness=request.moneyness,
                log_moneyness=request.log_moneyness,
                sqrt_time_to_expiry=request.sqrt_time_to_expiry,
                days_to_expiry=request.days_to_expiry,
                implied_volatility=request.implied_volatility,
                model_type=model_type,
            )
            response = await asyncio.wait_for(stub.Predict(grpc_req), timeout=0.1)  # 100ms timeout
            price = response.price
            source = f"grpc_{model_type}"
        except Exception as e:
            logger.warning("grpc_inference_failed_falling_back_to_bs", error=str(e))
            # 2. Computation Fallback (Black-Scholes)
            price = self._black_scholes_price(request)
            source = "black_scholes_fallback"

        duration = (time.perf_counter() - start_time) * 1000
        ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)

        # 3. Fire-and-forget High-Performance Persistence (Hypertable)
        asyncio.create_task(self._persist_prediction(symbol, price, request))

        return InferenceResponse(price=price, model_type=source, latency_ms=duration)

    async def _persist_prediction(self, symbol: str, price: float, request: InferenceRequest):
        """Asynchronously log prediction to the hypertable (Optimized)."""
        try:
            from datetime import UTC, datetime

            import msgspec

            input_features_json = msgspec.json.encode(request.model_dump()).decode()

            # Format for VectorizedDBEngine.insert_predictions_bulk
            # columns: (timestamp, symbol, model_id, input_features, predicted_price)
            prediction_data = [
                (
                    datetime.now(UTC),
                    symbol,
                    None,  # model_id
                    input_features_json,
                    price,
                )
            ]

            async with db_engine as db:
                await db.insert_predictions_bulk(prediction_data)

        except Exception as e:
            logger.error("prediction_persistence_failed", error=str(e))

    async def close(self):
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None


_ml_service_instance: MLService | None = None


def get_ml_service() -> MLService:
    """Returns the singleton MLService instance."""
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance