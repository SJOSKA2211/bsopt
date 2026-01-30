import onnxruntime as ort
import numpy as np
import structlog
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from src.shared.observability import ONNX_INFERENCE_LATENCY

logger = structlog.get_logger(__name__)

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    latency_ms: float

class ONNXModelServer:
    """
    Ultra-low latency model server using ONNX Runtime.
    Provides significantly faster inference than standard MLflow serving.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Optimize for performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Prioritize GPU if available
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("onnx_session_initialized", model_path=model_path, providers=providers)
        except Exception as e:
            logger.error("onnx_init_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Execute inference"""
        return self.session.run([self.output_name], {self.input_name: features.astype(np.float32)})[0]

from fastapi.responses import ORJSONResponse

# FastAPI Application for Serving
app = FastAPI(
    title="BS-Opt ONNX Serving",
    default_response_class=ORJSONResponse
)
model_server: ONNXModelServer = None

@app.on_event("startup")
async def load_model():
    global model_server
    model_path = os.getenv("ONNX_MODEL_PATH", "models/latest_pricing.onnx")
    if os.path.exists(model_path):
        model_server = ONNXModelServer(model_path)
    else:
        logger.warning("model_not_found_deferred_loading", path=model_path)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    start_time = time.perf_counter()
    try:
        X = np.array(request.features)
        preds = model_server.predict(X)
        latency = (time.perf_counter() - start_time) * 1000
        ONNX_INFERENCE_LATENCY.observe(latency)
        
        return PredictionResponse(
            predictions=preds.flatten().tolist(),
            latency_ms=latency
        )
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_server is not None}
