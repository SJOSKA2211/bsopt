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
    ml_service: MLService = Depends(get_ml_service)
) -> DataResponse:
    """Predict option price using ML models."""
    return DataResponse(data=await ml_service.predict(request, model_type))

@router.get("/drift-metrics")
async def get_drift_metrics(
    model_id: UUID | None = None,
    db: Session = Depends(get_db)
) -> DataResponse:
    """Fetch model performance metrics."""
    return DataResponse(
        data=DriftMetricsResponse(metrics=get_model_drift_metrics(db, model_id))
    )
