import asyncio
import logging
import time
import typing
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort

from services.config import settings
from services.protos import inference_pb2, inference_pb2_grpc

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

    def update_models(self, xgb_model, nn_ort_session):
        """High-Performance: Live-update model sessions without restart."""
        self.xgb_model = xgb_model
        self.nn_ort_session = nn_ort_session
        logger.info("grpc_models_updated")

    def _prepare_input(self, request) -> np.ndarray:
        """Consolidated and optimized input preparation."""
        # OPTIMIZED: Return a flat array, reshape only when needed
        return np.array(
            [
                request.underlying_price,
                request.strike,
                request.time_to_expiry,
                float(request.is_call),
                request.moneyness,
                request.log_moneyness,
                request.sqrt_time_to_expiry,
                request.days_to_expiry,
                request.implied_volatility,
            ],
            dtype=np.float32,
        )

    @typing.override
    async def Predict(self, request, context):
        start_time = time.perf_counter()
        model_type = request.model_type or "xgb"

        try:
            # 1. Resolve Session
            session = self.xgb_model if model_type == "xgb" else self.nn_ort_session
            if session is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(f"Model '{model_type}' not loaded")
                return inference_pb2.InferenceResponse()

            # 2. Optimized Input Prep
            input_data = self._prepare_input(request).reshape(1, -1)

            # 3. Fast-Path Inference (Direct ONNX execution)
            if isinstance(session, ort.InferenceSession):
                input_name = session.get_inputs()[0].name
                # Use named outputs or at least be explicit
                outputs = session.run(None, {input_name: input_data})
                prediction = outputs[0].flatten()[0]
            else:
                # Fallback for standard models
                prediction = session.predict(input_data)[0]

            latency_ms = (time.perf_counter() - start_time) * 1000

            return inference_pb2.InferenceResponse(
                price=float(prediction), model_type=model_type, latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"gRPC inference error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferenceResponse()


async def serve_grpc(xgb_model, nn_ort_session):
    """Starts the gRPC server and returns the servicer."""
    options = [
        ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ("grpc.default_compression_algorithm", grpc.Compression.Gzip),
        ("grpc.default_compression_level", grpc.CompressionLevel.High),
    ]
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    servicer = MLInferenceServicer(xgb_model, nn_ort_session)
    inference_pb2_grpc.add_MLInferenceServicer_to_server(servicer, server)

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

    # We return the servicer so the REST app can update its state
    return servicer


if __name__ == "__main__":
    # Example manual execution for testing
    logging.basicConfig(level=logging.INFO)

    # Mock models for standalone testing if needed
    xgb = None
    nn = None

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve_grpc(xgb, nn))
    except KeyboardInterrupt:
        pass
