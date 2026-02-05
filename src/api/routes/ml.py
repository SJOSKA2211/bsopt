"""
Machine Learning Routes (Singularity Refactored)
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.schemas.common import DataResponse
from src.api.schemas.ml import DriftMetricsResponse, InferenceRequest
from src.database import get_db
from src.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/ml", tags=["Machine Learning"])
logger = structlog.get_logger(__name__)

@router.post("/predict")
async def predict(
    request: InferenceRequest, 
    model_type: str = "xgb",
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Predict option price using ML models.
    """
    result = await ml_service.predict(request, model_type)
    return DataResponse(data=result)

@router.get("/drift-metrics")
async def get_drift_metrics(
    model_id: UUID | None = None,
    db: Session = Depends(get_db)
):
    """
    Fetch model performance metrics.
    """
    # Assuming get_model_drift_metrics is updated for sync session
    from src.database.crud import get_model_drift_metrics
    metrics = get_model_drift_metrics(db, model_id)
    return DataResponse(data=DriftMetricsResponse(metrics=metrics))