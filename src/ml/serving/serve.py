import logging
import os
import time

from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model Serving")

# Metrics
INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds", "Time spent processing prediction", ["model_type"]
)
PREDICTION_COUNT = Counter(
    "ml_predictions_total", "Total number of predictions", ["status", "model_type"]
)
MODEL_LOAD_STATUS = Gauge(
    "ml_model_load_status", "Status of model loading (1 for success, 0 for failure)"
)

# Global model state
state: Dict[str, Any] = {"xgb_model": None, "nn_ort_session": None, "current_model": "xgb"}


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    await load_xgb_model()
    await load_onnx_model()


async def load_xgb_model():
    try:
        model_uri = "models:/XGBoostOptionPricer/Production"
        logger.info(f"Loading XGBoost model from {model_uri}...")
        state["xgb_model"] = mlflow.pyfunc.load_model(model_uri)
        MODEL_LOAD_STATUS.set(1)
    except Exception as e:
        logger.error(f"XGBoost load failed: {e}")


async def load_onnx_model():
    try:
        # Load latest production ONNX if exists
        onnx_path = "models/nn_option_pricer.onnx"
        if os.path.exists(onnx_path):
            state["nn_ort_session"] = ort.InferenceSession(onnx_path)
            logger.info("ONNX session initialized.")
    except Exception as e:
        logger.error(f"ONNX load failed: {e}")


class InferenceRequest(BaseModel):
    underlying_price: float
    strike: float
    time_to_expiry: float
    is_call: int
    moneyness: float
    log_moneyness: float
    sqrt_time_to_expiry: float
    days_to_expiry: float
    implied_volatility: float


@app.post("/predict")
async def predict(request: InferenceRequest, model_type: str = "xgb"):
    start_time = time.time()
    try:
        if model_type == "xgb":
            if state["xgb_model"] is None:
                raise HTTPException(status_code=503, detail="XGB model not available")
            df = pd.DataFrame([request.dict()])
            prediction = state["xgb_model"].predict(df)[0]

        elif model_type == "nn":
            if state["nn_ort_session"] is None:
                raise HTTPException(status_code=530, detail="NN model not available")
            input_data = np.array(list(request.dict().values()), dtype=np.float32).reshape(1, -1)
            ort_inputs = {state["nn_ort_session"].get_inputs()[0].name: input_data}
            prediction = state["nn_ort_session"].run(None, ort_inputs)[0][0]

        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        latency = time.time() - start_time
        INFERENCE_LATENCY.labels(model_type=model_type).observe(latency)
        PREDICTION_COUNT.labels(status="success", model_type=model_type).inc()

        return {"price": float(prediction), "latency": latency}

    except Exception as e:
        PREDICTION_COUNT.labels(status="error", model_type=model_type).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "xgb_loaded": state["xgb_model"] is not None,
        "nn_loaded": state["nn_ort_session"] is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
