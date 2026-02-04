import onnxruntime as ort
import numpy as np
import structlog
import os
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Response
import msgspec
import time
from src.shared.observability import ONNX_INFERENCE_LATENCY

logger = structlog.get_logger(__name__)

# Use msgspec for high-performance validation and serialization
class PredictionRequest(msgspec.Struct):
    features: List[List[float]]

class PredictionResponse(msgspec.Struct):
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
        if 'TensorrtExecutionProvider' in available_providers:
            providers.append('TensorrtExecutionProvider')
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

# FastAPI Application for Serving
app = FastAPI(
    title="BS-Opt ONNX Serving"
)
model_server: ONNXModelServer = None

# Pre-compiled encoder/decoder for maximum performance
decoder = msgspec.json.Decoder(PredictionRequest)
encoder = msgspec.json.Encoder()

@app.on_event("startup")
async def load_model():
    global model_server
    # Prioritize quantized model if available
    quantized_path = os.getenv("ONNX_QUANTIZED_MODEL_PATH", "models/latest_pricing_int8.onnx")
    model_path = os.getenv("ONNX_MODEL_PATH", "models/latest_pricing.onnx")
    
    final_path = quantized_path if os.path.exists(quantized_path) else model_path
    
    if os.path.exists(final_path):
        logger.info("loading_onnx_model", path=final_path, is_quantized=(final_path == quantized_path))
        model_server = ONNXModelServer(final_path)
    else:
        logger.warning("model_not_found_deferred_loading", path=final_path)

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
        
        response_data = PredictionResponse(
            predictions=preds.flatten().tolist(),
            latency_ms=latency
        )
        
        # Fast serialization
        return Response(
            content=encoder.encode(response_data),
            media_type="application/json"
        )
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_server is not None}
