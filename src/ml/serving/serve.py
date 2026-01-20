import logging
import os
import time
import uuid
from datetime import datetime, timezone # Added timezone
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api.schemas.common import DataResponse, ErrorDetail, ErrorResponse
from src.api.schemas.ml import InferenceRequest, InferenceResponse
from src.utils.circuit_breaker import InMemoryCircuitBreaker, DistributedCircuitBreaker # Import both
from src.utils.cache import get_redis # Import for DistributedCircuitBreaker initialization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BSOPT ML Serving",
    version="1.0.0",
    description="Production-grade ML model serving for option pricing"
)

# Metrics
INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds", "Time spent processing prediction", ["model_type"]
)
PREDICTION_COUNT = Counter(
    "ml_predictions_total", "Total number of predictions", ["status", "model_type"]
)
MODEL_LOAD_STATUS = Gauge(
    "ml_model_load_status", "Status of model loading (1 for success, 0 for failure)", ["model_type"]
)

# Circuit Breakers
ml_circuit: InMemoryCircuitBreaker = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30) # Default to in-memory

# Global model state
state: Dict[str, Any] = {"xgb_model": None, "nn_ort_session": None, "current_model": "xgb"}


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    
    # Attempt to initialize DistributedCircuitBreaker if Redis is available
    global ml_circuit
    try:
        redis_client = get_redis()
        if redis_client is None:
            raise HTTPException(status_code=500, detail="Redis client not initialized for DistributedCircuitBreaker")

        ml_circuit = DistributedCircuitBreaker(
            name="ml_inference",
            redis_client=redis_client,
            failure_threshold=5,
            recovery_timeout=30
        )
        logger.info("Distributed Circuit Breaker initialized for ML inference.")
    except HTTPException:
        logger.warning("Redis client not available, falling back to in-memory circuit breaker for ML inference.")

    await load_xgb_model()
    await load_onnx_model()


async def load_xgb_model():
    try:
        model_uri = os.getenv("XGB_MODEL_URI", "models:/XGBoostOptionPricer/Production")
        logger.info(f"Loading XGBoost model from {model_uri}...")
        state["xgb_model"] = mlflow.pyfunc.load_model(model_uri)
        MODEL_LOAD_STATUS.labels(model_type="xgb").set(1)
    except Exception as e:
        logger.error(f"XGBoost load failed: {e}")
        MODEL_LOAD_STATUS.labels(model_type="xgb").set(0)


async def load_onnx_model():
    try:
        # Construct absolute path relative to the current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_onnx_path = os.path.join(base_dir, "models", "nn_option_pricer.onnx")
        onnx_path = os.getenv("NN_MODEL_PATH", default_onnx_path)
        
        if os.path.exists(onnx_path):
            state["nn_ort_session"] = ort.InferenceSession(onnx_path)
            logger.info(f"ONNX session initialized from {onnx_path}.")
            MODEL_LOAD_STATUS.labels(model_type="nn").set(1)
        else:
            logger.warning(f"ONNX model not found at {onnx_path}")
            MODEL_LOAD_STATUS.labels(model_type="nn").set(0)
    except Exception as e:
        logger.error(f"ONNX load failed: {e}")
        MODEL_LOAD_STATUS.labels(model_type="nn").set(0)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled exception [RID: {request_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred during inference",
                details=None,
                request_id=request_id,
            )
        ),
    )


@app.post(
    "/predict",
    response_model=DataResponse[InferenceResponse],
    responses={
        503: {"model": ErrorResponse},
        400: {"model": ErrorResponse}
    }
)
@ml_circuit
async def predict(request: InferenceRequest, model_type: str = "xgb"):
    """
    Perform ML-based option price prediction.
    - **xgb**: eXtreme Gradient Boosting model (Default)
    - **nn**: Deep Neural Network (ONNX)
    """
    start_time = time.perf_counter()
    
    try:
        if model_type == "xgb":
            if state["xgb_model"] is None:
                raise HTTPException(status_code=503, detail="XGB model currently unavailable")
            
            # Convert request to DF
            df = pd.DataFrame([request.model_dump()])
            prediction = state["xgb_model"].predict(df)[0]

        elif model_type == "nn":
            if state["nn_ort_session"] is None:
                raise HTTPException(status_code=503, detail="Neural Network model currently unavailable")
            
            input_data = np.array(list(request.model_dump().values()), dtype=np.float32).reshape(1, -1)
            ort_inputs = {state["nn_ort_session"].get_inputs()[0].name: input_data}
            prediction = state["nn_ort_session"].run(None, ort_inputs)[0][0]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        INFERENCE_LATENCY.labels(model_type=model_type).observe(latency_ms / 1000)
        PREDICTION_COUNT.labels(status="success", model_type=model_type).inc()

        return DataResponse(
            data=InferenceResponse(
                price=float(prediction),
                model_type=model_type,
                latency_ms=latency_ms
            )
        )

    except HTTPException:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc()
        raise
    except Exception as e:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc()
        logger.error(f"Inference processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "xgb": state["xgb_model"] is not None,
            "nn": state["nn_ort_session"] is not None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat() # Use timezone-aware datetime
    }


if __name__ == "__main__":
    import uvicorn

    from src.config import settings
    uvicorn.run(app, host=settings.ML_SERVICE_HOST, port=settings.ML_SERVICE_PORT)
