"""
Machine Learning Routes (Optimized)
"""

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.jwt_validator import require_tier
from api.responses import MsgspecJSONResponse
from api.schemas.common import DataResponse
from api.schemas.ml import (
    ComparisonMetrics,
    DriftMetricsResponse,
    InferenceRequest,
    InferenceResponse,
)
from src.auth.auth import get_current_active_user
from src.database import get_async_db
from src.database.crud import get_model_drift_metrics
from src.database.models import User
from src.ml.service import MLService, get_ml_service
from src.shared.utils.circuit_breaker import ml_client_circuit

router = APIRouter(
    prefix="/ml", tags=["Machine Learning"], default_response_class=MsgspecJSONResponse
)
logger = structlog.get_logger(__name__)


@router.get("/comparison", response_model=DataResponse[ComparisonMetrics])
async def get_ml_comparison(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[ComparisonMetrics]:
    """
    Fetch Production AI vs Human performance comparison.
    OPTIMIZED: Aggregates real metrics from the persistence layer.
    """
    from src.database.crud import get_ml_comparison_stats

    stats = await get_ml_comparison_stats(db, current_user.id)
    return DataResponse(data=ComparisonMetrics(**stats))


@router.post("/predict", response_model=None)
@ml_client_circuit
async def predict(
    request: InferenceRequest,
    symbol: str = "UNKNOWN",
    model_type: str = "xgb",
    current_user: User = Depends(get_current_active_user),
    ml_service: MLService = Depends(get_ml_service),
) -> DataResponse[InferenceResponse]:
    """Predict option price using ML models."""
    return DataResponse(data=await ml_service.predict(request, model_type, symbol))


@router.get("/predictions", response_model=DataResponse[InferenceResponse])
@ml_client_circuit
async def get_predictions(
    symbol: str | None = None,
    model_type: str = "xgb",
    current_user: User = Depends(get_current_active_user),
    ml_service: MLService = Depends(get_ml_service),
) -> DataResponse[InferenceResponse]:
    """
    Convenience endpoint for the frontend dashboard.
    """
    from src.shared.config import settings
    from src.shared.utils.sanitization import sanitize_alphanumeric

    if symbol is None:
        symbol = settings.MARKET_TICKER_SYMBOLS[0] if settings.MARKET_TICKER_SYMBOLS else "SPX"

    symbol = sanitize_alphanumeric(symbol.strip().upper())
    if not symbol or len(symbol) > 10:
        return DataResponse(data=None, message="Invalid symbol")

    base_price = 100.0
    req = InferenceRequest(
        underlying_price=base_price,
        strike=base_price,
        time_to_expiry=1.0,
        is_call=1,
        moneyness=1.0,
        log_moneyness=0.0,
        sqrt_time_to_expiry=1.0,
        days_to_expiry=365.0,
        implied_volatility=0.2,
    )
    return DataResponse(data=await ml_service.predict(req, model_type, symbol))


@router.get(
    "/drift-metrics",
    response_model=DataResponse[DriftMetricsResponse],
    dependencies=[Depends(require_tier(["admin", "enterprise"]))],
)
async def get_drift_metrics(
    model_id: UUID | None = None, db: AsyncSession = Depends(get_async_db)
) -> DataResponse[DriftMetricsResponse]:
    """Fetch model performance metrics (Async Optimized)."""
    # Note: CRUD method name assumed to be aligned with async pattern
    metrics = await get_model_drift_metrics(db, model_id)
    return DataResponse(data=DriftMetricsResponse(metrics=metrics))


@router.post(
    "/retrain",
    response_model=DataResponse[dict[str, str]],
    dependencies=[Depends(require_tier(["admin", "enterprise"]))],
)
async def trigger_retraining(
    ticker: str | None = None,
    force: bool = False,
    threshold: int = 50000,
    mode: str = "regressor",
) -> DataResponse[dict[str, str]]:
    """
    Trigger model retraining.
    Modes: 'regressor' (single ticker), 'cross_sectional' (entire universe).
    """
    from src.shared.config import settings

    if ticker is None:
        ticker = settings.MARKET_TICKER_SYMBOLS[0] if settings.MARKET_TICKER_SYMBOLS else "SPX"

    from src.workers.tasks.ml_tasks import check_threshold_and_retrain_task

    task = check_threshold_and_retrain_task.delay(
        ticker=ticker, force=force, threshold=threshold, mode=mode
    )
    return DataResponse(
        data={"task_id": task.id, "status": "dispatched", "mode": mode},
        message=f"Retraining task ({mode}) dispatched to background worker",
    )


@router.get("/health")
async def ml_health() -> dict[str, Any]:
    """
    ML Integrated Health Mesh Endpoint.
    Consolidates MLflow, Prometheus, and Redis Anomaly metrics.
    """
    import os
    from datetime import UTC, datetime

    # Mock data if allowed
    if os.getenv("BSOPT_ALLOW_WEAK_SECRETS") == "1":
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "mlflow": {
                "stage": "Production",
                "drift_detected": False,
                "last_run_id": "simulated_run_001",
            },
            "prometheus": {
                "error_rate_5xx": 0.0,
                "p95_latency": 12.5,
                "cpu_usage": 0.45,
                "memory_usage": 512 * 1024 * 1024,
            },
            "redis_anomalies": [],
        }

    # Real implementation placeholder (logic to be fleshed out in Phase 3)
    return {
        "status": "unhealthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "error": "Real ML metrics extraction not yet implemented",
    }
