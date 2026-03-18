import logging
import os
import time
import uuid
from datetime import UTC, datetime  # Added timezone
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.responses import MsgspecJSONResponse, Response
from src.api.schemas.common import DataResponse
from src.api.schemas.ml import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    InferenceRequest,
    InferenceResponse,
)
from src.ml.utils.inference import ONNXInferenceEngine
from src.shared.utils.circuit_breaker import (  # Import both
    DistributedCircuitBreaker,
    InMemoryCircuitBreaker,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="BSOPT ML Serving",
    version="1.0.0",
    description="Production-grade ML model serving for option pricing",
    default_response_class=MsgspecJSONResponse,
)

# Metrics
INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds", "Time spent processing prediction", ["model_type"]
)
PREDICTION_COUNT = Counter(
    "ml_predictions_total", "Total number of predictions", ["status", "model_type"]
)
MODEL_LOAD_STATUS = Gauge(
    "ml_model_load_status",
    "Status of model loading (1 for success, 0 for failure)",
    ["model_type"],
)

# Circuit Breakers
ml_circuit: InMemoryCircuitBreaker = InMemoryCircuitBreaker(
    failure_threshold=5, recovery_timeout=30
)  # Default to in-memory

# Global model state
state: dict[str, Any] = {
    "xgb_model": None,
    "xgb_ort_session": None,
    "nn_ort_session": None,
    "current_model": "xgb",
    "grpc_servicer": None,
}


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(settings.tracking_uri)

    # Attempt to initialize DistributedCircuitBreaker if Redis is available
    global ml_circuit
    try:
        from src.shared.utils.cache import _redis_pool

        redis_client = _redis_pool

        if redis_client is not None:
            ml_circuit = DistributedCircuitBreaker(
                name="ml_inference",
                redis_client=redis_client,
                failure_threshold=5,
                recovery_timeout=30,
            )
            logger.info("Distributed Circuit Breaker initialized for ML inference.")
        else:
            logger.warning(
                "Redis pool not available, using in-memory circuit breaker for ML inference."
            )
    except Exception as e:
        logger.warning(
            f"Failed to initialize distributed circuit breaker: {e}. Falling back to in-memory."
        )

    await load_xgb_model()
    await load_onnx_model()

    # Start gRPC server in background
    from src.ml.serving.grpc_server import serve_grpc

    servicer = await serve_grpc(
        (state["xgb_model"] if state["xgb_ort_session"] is None else state["xgb_ort_session"]),
        state["nn_ort_session"],
    )
    state["grpc_servicer"] = servicer


async def load_xgb_model():
    """Load XGBoost model, favoring quantized ONNX for maximum performance."""
    import anyio

    try:
        # Check for Quantized ONNX first
        int8_path = os.getenv("XGB_INT8_MODEL_PATH", "models/latest_xgb_pricing.int8.onnx")
        exists_int8 = await anyio.to_thread.run_sync(os.path.exists, int8_path)
        if exists_int8:
            state["xgb_ort_session"] = ONNXInferenceEngine(int8_path)
            logger.info(f"XGBoost INT8 Quantized engine initialized from {int8_path}.")
            MODEL_LOAD_STATUS.labels(model_type="xgb_int8").set(1)
            return

        # Fallback to standard ONNX
        onnx_path = os.getenv("XGB_ONNX_MODEL_PATH", "models/latest_xgb_pricing.onnx")
        exists_onnx = await anyio.to_thread.run_sync(os.path.exists, onnx_path)
        if exists_onnx:
            state["xgb_ort_session"] = ONNXInferenceEngine(onnx_path)
            logger.info(f"XGBoost ONNX engine initialized from {onnx_path}.")
            MODEL_LOAD_STATUS.labels(model_type="xgb_onnx").set(1)
            return

        model_uri = os.getenv("XGB_MODEL_URI", "models:/XGBoostOptionPricer/Production")
        logger.info(f"Loading XGBoost model from {model_uri} via MLflow...")
        state["xgb_model"] = mlflow.pyfunc.load_model(model_uri)
        MODEL_LOAD_STATUS.labels(model_type="xgb").set(1)
    except Exception as e:
        logger.error(f"XGBoost load failed: {e}")
        MODEL_LOAD_STATUS.labels(model_type="xgb").set(0)


async def load_onnx_model():
    import anyio

    try:
        # Construct absolute path relative to the current file
        abs_path = await anyio.to_thread.run_sync(os.path.abspath, __file__)
        base_dir = os.path.dirname(abs_path)
        default_onnx_path = os.path.join(base_dir, "models", "nn_option_pricer.onnx")
        onnx_path = os.getenv("NN_MODEL_PATH", default_onnx_path)

        exists = await anyio.to_thread.run_sync(os.path.exists, onnx_path)
        if exists:
            state["nn_ort_session"] = ONNXInferenceEngine(onnx_path)
            logger.info(f"ONNX NN engine initialized from {onnx_path}.")
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
    return MsgspecJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during inference",
            "details": None,
            "request_id": request_id,
        },
    )


@app.post(
    "/predict",
    response_model=None,
)
@ml_circuit
async def predict(request: InferenceRequest, model_type: str = "xgb") -> DataResponse:
    """
    Perform ML-based option price prediction.
    - **xgb**: eXtreme Gradient Boosting model (Default, ONNX-accelerated if available)
    - **nn**: Deep Neural Network (ONNX)
    """
    start_time = time.perf_counter()

    try:
        if model_type == "xgb":
            if state["xgb_ort_session"]:
                input_data = np.array(
                    [
                        request.underlying_price,
                        request.strike,
                        request.time_to_expiry,
                        float(request.is_call),
                        request.moneyness,
                        request.log_moneyness,
                        request.sqrt_time_to_expiry,
                        request.days_to_expiry,
                        request.implied_volatility or 0.25,
                    ],
                    dtype=np.float32,
                ).reshape(1, -1)
                prediction = state["xgb_ort_session"].predict(input_data)[0][0]
            elif state["xgb_model"]:
                import msgspec

                df = pd.DataFrame([msgspec.to_builtins(request)])
                prediction = state["xgb_model"].predict(df)[0]
            else:
                raise HTTPException(status_code=503, detail="XGB model currently unavailable")

        elif model_type == "nn":
            if state["nn_ort_session"] is None:
                raise HTTPException(
                    status_code=503, detail="Neural Network model currently unavailable"
                )

            input_data = np.array(
                [
                    request.underlying_price,
                    request.strike,
                    request.time_to_expiry,
                    float(request.is_call),
                    request.moneyness,
                    request.log_moneyness,
                    request.sqrt_time_to_expiry,
                    request.days_to_expiry,
                    request.implied_volatility or 0.25,
                ],
                dtype=np.float32,
            ).reshape(1, -1)
            prediction = state["nn_ort_session"].predict(input_data)[0][0]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        latency_ms = (time.perf_counter() - start_time) * 1000
        INFERENCE_LATENCY.labels(model_type=model_type).observe(latency_ms / 1000)
        PREDICTION_COUNT.labels(status="success", model_type=model_type).inc()

        return DataResponse(
            data=InferenceResponse(
                price=float(prediction), model_type=model_type, latency_ms=latency_ms
            )
        )

    except HTTPException:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc()
        raise
    except Exception as e:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc()
        logger.error(f"Inference processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error") from e


@app.post(
    "/predict/batch",
    response_model=None,
)
@ml_circuit
async def predict_batch(request: BatchInferenceRequest, model_type: str = "xgb") -> DataResponse:
    """
    Perform batch ML-based option price prediction.
    - **xgb**: eXtreme Gradient Boosting model (Default, ONNX-accelerated if available)
    - **nn**: Deep Neural Network (ONNX)
    """
    start_time = time.perf_counter()

    try:
        # Optimization: Pre-allocate numpy array for zero-copy batch construction
        n_reqs = len(request.requests)
        input_data = np.empty((n_reqs, 9), dtype=np.float32)

        for i, r in enumerate(request.requests):
            input_data[i, 0] = r.underlying_price
            input_data[i, 1] = r.strike
            input_data[i, 2] = r.time_to_expiry
            input_data[i, 3] = float(r.is_call)
            input_data[i, 4] = r.moneyness
            input_data[i, 5] = r.log_moneyness
            input_data[i, 6] = r.sqrt_time_to_expiry
            input_data[i, 7] = r.days_to_expiry
            input_data[i, 8] = r.implied_volatility or 0.25

        if model_type == "xgb":
            if state["xgb_ort_session"]:
                preds = state["xgb_ort_session"].predict(input_data)
                predictions = [float(p[0]) for p in preds]
            elif state["xgb_model"]:
                # Fallback for non-ONNX models (slower)
                cols = [
                    "underlying_price",
                    "strike",
                    "time_to_expiry",
                    "is_call",
                    "moneyness",
                    "log_moneyness",
                    "sqrt_time_to_expiry",
                    "days_to_expiry",
                    "implied_volatility",
                ]
                df = pd.DataFrame(input_data, columns=cols)
                preds = state["xgb_model"].predict(df)
                predictions = [float(p) for p in preds]
            else:
                raise HTTPException(status_code=503, detail="XGB model currently unavailable")

        elif model_type == "nn":
            if state["nn_ort_session"] is None:
                raise HTTPException(
                    status_code=503, detail="Neural Network model currently unavailable"
                )

            preds = state["nn_ort_session"].predict(input_data)
            predictions = [float(p[0]) for p in preds]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_latency = total_latency_ms / len(predictions)

        # Log metrics once for the batch to save overhead
        INFERENCE_LATENCY.labels(model_type=model_type).observe(total_latency_ms / 1000)
        PREDICTION_COUNT.labels(status="success", model_type=model_type).inc(len(predictions))

        response_items = [
            InferenceResponse(
                price=p,
                model_type=model_type,
                latency_ms=avg_latency,
            )
            for p in predictions
        ]

        return DataResponse(
            data=BatchInferenceResponse(
                predictions=response_items, total_latency_ms=total_latency_ms
            )
        )

    except HTTPException:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc(len(request.requests))
        raise
    except Exception as e:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc(len(request.requests))
        logger.error(f"Batch inference processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal batch inference error") from e


@app.post("/ml/reload")
async def reload_models(request: Request):
    """
    Trigger a reload of models from the registry.
    """
    try:
        # Re-initialize models
        await load_xgb_model()
        await load_onnx_model()

        # Live-update gRPC servicer
        if state["grpc_servicer"]:
            state["grpc_servicer"].update_models(
                (
                    state["xgb_model"]
                    if state["xgb_ort_session"] is None
                    else state["xgb_ort_session"]
                ),
                state["nn_ort_session"],
            )

        return {"status": "reloaded", "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


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
        "timestamp": datetime.now(UTC).isoformat(),  # Use timezone-aware datetime
    }


if __name__ == "__main__":
    import uvicorn

    from src.config import settings

    uvicorn.run(app, host=settings.ML_SERVICE_HOST, port=settings.ML_SERVICE_PORT)
