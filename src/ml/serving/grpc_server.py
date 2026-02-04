import asyncio
import logging
import time
import grpc
from concurrent import futures
import numpy as np
import pandas as pd
import onnxruntime as ort
import mlflow.pyfunc
from typing import Any, Dict

from src.protos import inference_pb2, inference_pb2_grpc
from src.config import settings

logger = logging.getLogger(__name__)

class MLInferenceServicer(inference_pb2_grpc.MLInferenceServicer):
    """
    gRPC Servicer for high-performance ML inference.
    Handles requests for XGBoost and ONNX models.
    """
    def __init__(self, xgb_model, nn_ort_session):
        # xgb_model can be an ort.InferenceSession or a standard XGB model
        self.xgb_model = xgb_model
        self.nn_ort_session = nn_ort_session

    async def Predict(self, request, context):
        start_time = time.perf_counter()
        model_type = request.model_type or "xgb"
        
        try:
            if model_type == "xgb":
                if self.xgb_model is None:
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details("XGB model not loaded")
                    return inference_pb2.InferenceResponse()
                
                # Input data
                input_data = np.array([[
                    request.underlying_price, request.strike, request.time_to_expiry,
                    float(request.is_call), request.moneyness, request.log_moneyness,
                    request.sqrt_time_to_expiry, request.days_to_expiry, request.implied_volatility
                ]], dtype=np.float32)

                if isinstance(self.xgb_model, ort.InferenceSession):
                    # ONNX path (Task 1: reduce latency)
                    ort_inputs = {self.xgb_model.get_inputs()[0].name: input_data}
                    prediction = self.xgb_model.run(None, ort_inputs)[0][0]
                else:
                    # Standard XGB model path
                    prediction = self.xgb_model.predict(input_data)[0]

            elif model_type == "nn":
                if self.nn_ort_session is None:
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details("Neural Network model not loaded")
                    return inference_pb2.InferenceResponse()
                
                # Prepare input for ONNX
                input_data = np.array([
                    request.underlying_price, request.strike, request.time_to_expiry,
                    request.is_call, request.moneyness, request.log_moneyness,
                    request.sqrt_time_to_expiry, request.days_to_expiry, request.implied_volatility
                ], dtype=np.float32).reshape(1, -1)
                
                ort_inputs = {self.nn_ort_session.get_inputs()[0].name: input_data}
                prediction = self.nn_ort_session.run(None, ort_inputs)[0][0][0]
            
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Unsupported model type: {model_type}")
                return inference_pb2.InferenceResponse()

            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return inference_pb2.InferenceResponse(
                price=float(prediction),
                model_type=model_type,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"gRPC inference error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferenceResponse()

async def serve_grpc(xgb_model, nn_ort_session):
    """Starts the gRPC server."""
    options = [
        ('grpc.max_send_message_length', 16 * 1024 * 1024),
        ('grpc.max_receive_message_length', 16 * 1024 * 1024),
        ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
        ('grpc.default_compression_level', grpc.CompressionLevel.High),
    ]
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=options
    )
    inference_pb2_grpc.add_MLInferenceServicer_to_server(
        MLInferenceServicer(xgb_model, nn_ort_session), server
    )
    
    # Use the configured gRPC URL
    listen_addr = settings.ML_SERVICE_GRPC_URL
    # Ensure it's in the format expected by gRPC (e.g., [::]:50051)
    if ":" in listen_addr and not listen_addr.startswith("["):
        host, port = listen_addr.split(":")
        if host == "localhost":
            listen_addr = f"0.0.0.0:{port}"
            
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    # Example manual execution for testing
    import os
    logging.basicConfig(level=logging.INFO)
    
    # Mock models for standalone testing if needed
    xgb = None 
    nn = None
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve_grpc(xgb, nn))
    except KeyboardInterrupt:
        pass
