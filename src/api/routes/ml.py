import structlog
from fastapi import APIRouter, Depends
from typing import Optional
from uuid import UUID
from src.api.schemas.ml import InferenceRequest, InferenceResponse, DriftMetricsResponse
from src.api.schemas.common import DataResponse, ErrorResponse
from src.security.rate_limit import rate_limit
from src.services.ml_service import get_ml_service, MLService
from src.database import get_async_db
from src.database.crud import get_model_drift_metrics
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/ml", tags=["Machine Learning"], dependencies=[Depends(rate_limit)])
logger = structlog.get_logger(__name__)

@router.post(
    "/predict",
    response_model=DataResponse[InferenceResponse],
    responses={
        503: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def proxy_predict(
    request: InferenceRequest, 
    model_type: str = "xgb",
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Predict option price using machine learning models.
    Delegates to MLService for optimized execution and error handling.
    """
    result = await ml_service.predict(request, model_type)
    return DataResponse(data=result)

@router.get(
    "/drift-metrics",
    response_model=DataResponse[DriftMetricsResponse],
    responses={500: {"model": ErrorResponse}}
)
async def get_drift_metrics(
    model_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Fetch historical drift and performance metrics for ML models.
    Data is pre-aggregated hourly via materialized views.
    """
    metrics = await get_model_drift_metrics(db, model_id)
    return DataResponse(data=DriftMetricsResponse(metrics=metrics))
