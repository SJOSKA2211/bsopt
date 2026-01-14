import logging
import httpx
from fastapi import APIRouter, Depends, status
from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.api.schemas.common import DataResponse, ErrorResponse
from src.security.rate_limit import rate_limit
from src.config import settings
from src.api.exceptions import (
    ServiceUnavailableException,
    ValidationException,
    InternalServerException,
    BaseAPIException
)

router = APIRouter(prefix="/ml", tags=["Machine Learning"], dependencies=[Depends(rate_limit)])
logger = logging.getLogger(__name__)

ML_SERVICE_URL = settings.ML_SERVICE_URL # Ensure this is in settings

@router.post(
    "/predict",
    response_model=DataResponse[InferenceResponse],
    responses={
        503: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def proxy_predict(request: InferenceRequest, model_type: str = "xgb"):
    """
    Predict option price using machine learning models.
    Proxies request to the specialized ML serving microservice.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json=request.model_dump(),
                params={"model_type": model_type}
            )
            
            if response.status_code != 200:
                # Extract error from ML service if possible
                error_data = {}
                try:
                    error_data = response.json()
                except Exception:
                    logger.warning("Could not parse error response from ML service as JSON")
                
                detail = error_data.get("message", "ML service error")
                
                if response.status_code == 400:
                    raise ValidationException(message=detail)
                elif response.status_code == 503:
                    raise ServiceUnavailableException(message=detail)
                else:
                    raise InternalServerException(message=detail)
            
            return response.json()
            
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to ML service: {e}")
            raise ServiceUnavailableException(
                message="ML prediction service is currently unreachable"
            )
        except BaseAPIException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ML proxy: {e}")
            raise InternalServerException(message="Internal error during ML prediction")
