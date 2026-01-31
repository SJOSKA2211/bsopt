import httpx
import structlog
import time
from typing import Dict, Any, Optional
from src.config import settings
from src.utils.http_client import HttpClientManager
from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.api.exceptions import (
    ServiceUnavailableException,
    ValidationException,
    InternalServerException
)
from src.shared.observability import ML_PROXY_PREDICT_LATENCY
from src.utils.cache import multi_layer_cache

logger = structlog.get_logger(__name__)

class MLService:
    """
    Service for interacting with the Machine Learning serving layer.
    Manages connection pooling and provides robust error handling.
    """
    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL

    @multi_layer_cache(prefix="ml_predict", ttl=300, validation_model=InferenceResponse)
    async def predict(self, request: InferenceRequest, model_type: str = "xgb") -> InferenceResponse:
        """
        Proxies prediction request to the ML microservice with instrumentation.
        """
        # If the decorator returns a dict (from L2 cache), we need to validate it
        # This is a bit tricky with decorators. 
        # Alternatively, we can let the decorator handle the validation if we pass the return type.
        
        start_time = time.perf_counter()
        client = HttpClientManager.get_client()
        try:
            response = await client.post(
                f"{self.base_url}/predict",
                json=request.model_dump(),
                params={"model_type": model_type}
            )
            
            duration = time.perf_counter() - start_time
            ML_PROXY_PREDICT_LATENCY.observe(duration)

            if response.status_code != 200:
                self._handle_error_response(response)
            
            # Use Pydantic validation for external service response
            return InferenceResponse(**response.json())
            
        except httpx.RequestError as e:
            logger.error("ml_service_connection_failed", error=str(e))
            raise ServiceUnavailableException(
                message="ML prediction service is currently unreachable"
            )
        except Exception as e:
            if isinstance(e, (ValidationException, ServiceUnavailableException, InternalServerException)):
                raise
            logger.error("ml_service_unexpected_error", error=str(e))
            raise InternalServerException(message="Internal error during ML prediction")

    def _handle_error_response(self, response: httpx.Response):
        """Standardized error handling for ML service responses."""
        import orjson
        error_data = {}
        try:
            error_data = orjson.loads(response.content)
        except Exception:  # nosec
            pass
        
        detail = error_data.get("message", "ML service error")
        
        if response.status_code == 400:
            raise ValidationException(message=detail)
        elif response.status_code == 503:
            raise ServiceUnavailableException(message=detail)
        else:
            raise InternalServerException(message=detail)

# Singleton instance
ml_service = MLService()

def get_ml_service() -> MLService:
    return ml_service
