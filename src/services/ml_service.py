"""
Machine Learning Service
========================

Handles ML inference via gRPC or high-speed Shared Memory (SMI).
"""

import time
from math import erf, log, sqrt

import structlog

from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.shared.observability import ML_PROXY_PREDICT_LATENCY

logger = structlog.get_logger(__name__)


class MLService:
    """
    Local, lightweight ML pricing service.

    For development and demo purposes we avoid the external gRPC dependency and
    instead compute a simple Black–Scholes-style price approximation directly
    in-process. This keeps the API responsive even when the dedicated ML
    infrastructure is not running.
    """

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using math.erf."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _black_scholes_price(self, req: InferenceRequest) -> float:
        # Basic Black–Scholes approximation with fixed risk‑free rate and
        # using either provided implied_volatility or a sane default.
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
            # Fall back to a simple scaled intrinsic value if inputs are odd.
            intrinsic = max(S - K, 0.0) if req.is_call else max(K - S, 0.0)
            return intrinsic * 0.9

        if req.is_call:
            return S * Nd1 - K * (2.718281828459045 ** (-r * T)) * Nd2
        intrinsic = K * (2.718281828459045 ** (-r * T)) * (1 - Nd2) - S * (1 - Nd1)
        return max(intrinsic, 0.0)

    async def predict(
        self, request: InferenceRequest, model_type: str = "xgb"
    ) -> InferenceResponse:
        start_time = time.perf_counter()
        price = self._black_scholes_price(request)
        duration = (time.perf_counter() - start_time) * 1000
        ML_PROXY_PREDICT_LATENCY.observe(duration / 1000)

        return InferenceResponse(price=price, model_type=model_type, latency_ms=duration)

    async def close(self):
        # No external resources to close in the local implementation.
        return None


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
