"""
Machine Learning Routes (Optimized)
"""

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.responses import MsgspecJSONResponse
from src.api.schemas.common import DataResponseStruct
from src.api.schemas.ml import DriftMetricsResponse, InferenceRequest
from src.auth.auth import get_current_active_user, require_tier
from src.database import get_async_db
from src.database.crud import get_model_drift_metrics
from src.database.models import User
from src.ml.service import MLService, get_ml_service
from src.shared.utils.circuit_breaker import ml_client_circuit

router = APIRouter(
    prefix="/ml", tags=["Machine Learning"], default_response_class=MsgspecJSONResponse
)
logger = structlog.get_logger(__name__)


@router.post("/predict", response_model=None)
@ml_client_circuit
async def predict(
    request: InferenceRequest,
    symbol: str = "UNKNOWN",
    model_type: str = "xgb",
    current_user: User = Depends(get_current_active_user),
    ml_service: MLService = Depends(get_ml_service),
) -> Any:
    """Predict option price using ML models."""
    return DataResponseStruct(data=await ml_service.predict(request, model_type, symbol))


@router.get("/predictions", response_model=None)
@ml_client_circuit
async def get_predictions(
    symbol: str = "AAPL",
    model_type: str = "xgb",
    current_user: User = Depends(get_current_active_user),
    ml_service: MLService = Depends(get_ml_service),
) -> Any:
    """
    Convenience endpoint for the frontend dashboard.
    """
    from src.shared.utils.sanitization import sanitize_alphanumeric

    symbol = sanitize_alphanumeric(symbol.strip().upper())
    if not symbol or len(symbol) > 10:
        return DataResponseStruct(data={}, message="Invalid symbol")

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
    return DataResponseStruct(data=await ml_service.predict(req, model_type, symbol))


@router.get("/drift-metrics", response_model=None, dependencies=[Depends(require_tier(["admin", "enterprise"]))])
async def get_drift_metrics(
    model_id: UUID | None = None, db: AsyncSession = Depends(get_async_db)
) -> Any:
    """Fetch model performance metrics (Async Optimized)."""
    # Note: CRUD method name assumed to be aligned with async pattern
    metrics = await get_model_drift_metrics(db, model_id)
    return DataResponseStruct(data=DriftMetricsResponse(metrics=metrics))


@router.post("/retrain", response_model=None, dependencies=[Depends(require_tier(["admin", "enterprise"]))])
async def trigger_retraining(
    ticker: str = "AAPL",
    force: bool = False,
    threshold: int = 50000,
    mode: str = "regressor",
) -> Any:
    """
    Trigger model retraining.
    Modes: 'regressor' (single ticker), 'cross_sectional' (entire universe).
    """
    from src.workers.tasks.ml_tasks import check_threshold_and_retrain_task

    task = check_threshold_and_retrain_task.delay(
        ticker=ticker, force=force, threshold=threshold, mode=mode
    )
    return DataResponseStruct(
        data={"task_id": task.id, "status": "dispatched", "mode": mode},
        message=f"Retraining task ({mode}) dispatched to background worker",
    )
