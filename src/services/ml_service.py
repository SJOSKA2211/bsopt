"""
Machine Learning Service
Enhanced with God-Mode Persistence and Vectorized Database Ingestion.
"""

import time
import asyncio
from math import erf, log, sqrt
from typing import Any

import structlog

from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.shared.observability import ML_PROXY_PREDICT_LATENCY
from src.database.pipeliner import db_engine

logger = structlog.get_logger(__name__)


class MLService:
    """
    Enhanced ML pricing service with automated hypertable persistence.
    """

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using math.erf."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _black_scholes_price(self, req: InferenceRequest) -> float:
        # Basic Black–Scholes approximation
        S = req.underlying_price
        K = req.strike
        T = max(req.time_to_expiry, 1e-6)
        r = 0.01
        sigma = req.implied_volatility or 0.25

        try:
            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            Nd1 = self._norm_cdf(d1)
            Nd2 = self._norm_cdf(d2)
        except Exception:
            intrinsic = max(S - K, 0.0) if req.is_call else max(K - S, 0.0)
            return intrinsic * 0.9

        if req.is_call:
            return S * Nd1 - K * (2.718281828459045 ** (-r * T)) * Nd2
        intrinsic = K * (2.718281828459045 ** (-r * T)) * (1 - Nd2) - S * (1 - Nd1)
        return max(intrinsic, 0.0)

    async def predict(
        self, request: InferenceRequest, model_type: str = "xgb", symbol: str = "UNKNOWN"
    ) -> InferenceResponse:
        start_time = time.perf_counter()
        
        # 1. Computation
        price = self._black_scholes_price(request)
        duration = (time.perf_counter() - start_time) * 1000
        ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)

        # 2. Fire-and-forget God-Mode Persistence (Hypertable)
        # Optimized: uses VectorizedDBEngine with Binary COPY
        asyncio.create_task(self._persist_prediction(symbol, price, request))

        return InferenceResponse(price=price, model_type=model_type, latency_ms=duration)

    async def _persist_prediction(self, symbol: str, price: float, request: InferenceRequest):
        """Asynchronously log prediction to the hypertable."""
        try:
            from datetime import datetime, UTC
            import json
            
            # Format for VectorizedDBEngine.insert_predictions_bulk
            # columns: (timestamp, symbol, model_id, input_features, predicted_price)
            prediction_data = [
                (
                    datetime.now(UTC),
                    symbol,
                    None, # model_id (NULL for BS fallback)
                    json.dumps(request.model_dump()),
                    price
                )
            ]
            
            async with db_engine as db:
                await db.insert_predictions_bulk(prediction_data)
                
        except Exception as e:
            logger.error("prediction_persistence_failed", error=str(e))

    async def close(self):
        return None


_ml_service_instance: MLService | None = None


def get_ml_service() -> MLService:
    """Returns the singleton MLService instance."""
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLService()
    return _ml_service_instance
