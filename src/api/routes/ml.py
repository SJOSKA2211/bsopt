"""
Machine Learning Routes (Optimized)
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas.common import DataResponse
from src.api.schemas.ml import DriftMetricsResponse, InferenceRequest
from src.database import get_db
from src.database.crud import get_model_drift_metrics
from src.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
logger = structlog.get_logger(__name__)


@router.post("/predict")
async def predict(
    request: InferenceRequest,
    model_type: str = "xgb",
    ml_service: MLService = Depends(get_ml_service),
) -> DataResponse:
    """Predict option price using ML models."""
    return DataResponse(data=await ml_service.predict(request, model_type))


@router.get("/predictions")
async def get_predictions(
    symbol: str = "AAPL",
    model_type: str = "xgb",
    ml_service: MLService = Depends(get_ml_service),
) -> DataResponse:
    """
    Convenience endpoint for the frontend dashboard.

    The UI calls ``GET /api/v1/ml/predictions?symbol=...`` with a symbol only;
    here we synthesize a reasonable ``InferenceRequest`` using that symbol and
    delegate to the main ``/predict`` logic.
    """
    symbol = symbol.strip().upper()
    if not symbol.isalnum() or len(symbol) > 10:
        return DataResponse(data={}, message="Invalid symbol")

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
    return DataResponse(data=await ml_service.predict(req, model_type))


@router.get("/drift-metrics")
async def get_drift_metrics(
    model_id: UUID | None = None, db: Session = Depends(get_db)
) -> DataResponse:
    """Fetch model performance metrics."""
    return DataResponse(data=DriftMetricsResponse(metrics=get_model_drift_metrics(db, model_id)))
