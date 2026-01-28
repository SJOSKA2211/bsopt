import structlog
from fastapi import APIRouter, Depends
from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.api.schemas.common import DataResponse, ErrorResponse
from src.security.rate_limit import rate_limit
from src.services.ml_service import get_ml_service, MLService

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
