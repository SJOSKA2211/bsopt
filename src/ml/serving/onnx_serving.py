import os
import time
from contextlib import asynccontextmanager

import msgspec
import numpy as np
import onnxruntime as ort
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_fastapi_instrumentator import Instrumentator
from src.ml.serving.health import get_serving_health

from src.shared.observability import ONNX_INFERENCE_LATENCY

logger = structlog.get_logger(__name__)

# Use msgspec for high-performance validation and serialization
class PredictionRequest(msgspec.Struct):
    features: list[list[float]]

class PredictionResponse(msgspec.Struct):
    predictions: list[float]
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
        
        sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

        # Prioritize GPU if available
        available_providers = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("onnx_session_initialized", model_path=model_path, providers=providers)
        except Exception as e:
            logger.error("onnx_init_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Execute inference with explicit float32 casting."""
        # Ensure correct type for the engine
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: features})[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan manager for model resources.
    """
    global model_server
    import anyio

    # Prioritize quantized model
    model_path = os.getenv("ONNX_MODEL_PATH", "models/latest_pricing.onnx")
    if await anyio.to_thread.run_sync(os.path.exists, model_path):
        logger.info("loading_onnx_model", path=model_path)
        model_server = ONNXModelServer(model_path)

    yield
    # Cleanup logic if needed
    model_server = None

# Module-level state
model_server: ONNXModelServer | None = None
decoder = msgspec.json.Decoder(PredictionRequest)
encoder = msgspec.json.Encoder()

app = FastAPI(title="BS-Opt ONNX Serving", lifespan=lifespan)

# Instrument for Prometheus
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
async def predict(raw_request: Request):
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")

    start_time = time.perf_counter()
    try:
        # High-performance parsing
        body = await raw_request.body()
        request = decoder.decode(body)

        X = np.array(request.features)
        preds = model_server.predict(X)
        latency = (time.perf_counter() - start_time) * 1000
        ONNX_INFERENCE_LATENCY.observe(latency)

        response_data = PredictionResponse(predictions=preds.flatten().tolist(), latency_ms=latency)

        # Fast serialization
        return Response(content=encoder.encode(response_data), media_type="application/json")
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/health/liveness")
async def liveness():
    """Basic process check."""
    return {"status": "alive"}

@app.get("/health/readiness")
async def readiness():
    """Deep check for model loading and MLflow connectivity."""
    health_data = get_serving_health()
    if health_data["status"] != "healthy":
        from fastapi import Response
        return Response(content=str(health_data), status_code=503)
    return health_data

@app.get("/health")
async def legacy_health():
    """Backward compatibility health endpoint."""
    return get_serving_health()
