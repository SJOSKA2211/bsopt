import logging
import time
from typing import cast

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """
    High-performance inference engine using ONNX Runtime.
    Optimized for pure CPU execution with graph optimizations.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None):
        self.model_path = model_path

        # 1. Optimize Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

        sess_options.intra_op_num_threads = 1  # Standard for multi-worker ASGI
        sess_options.inter_op_num_threads = 2  # Balance between latency and concurrency

        # 2. Select Providers
        if providers is None:
            # Force CPU execution for lightweight, pure-CPU architecture
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"onnx_session_ready at {model_path} with providers {providers}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on batch of features.
        """
        start_time = time.perf_counter()

        # Ensure features are float32 for ONNX
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: features})

        latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"ONNX inference latency: {latency:.2f}ms")

        return cast(np.ndarray, outputs[0])


def optimize_onnx_model(input_path: str, output_path: str):
    """
    Apply ONNX Runtime optimizations to a model file.
    Uses ORT's internal graph optimization levels.
    """
    logger.info(f"Optimizing ONNX model from {input_path} to {output_path}")

    try:
        # Configure optimization options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_path

        # Creating a session with optimized_model_filepath set will trigger optimization and save the file
        # We use the 'CPUExecutionProvider' for general compatibility during optimization
        ort.InferenceSession(input_path, sess_options, providers=["CPUExecutionProvider"])

        logger.info(f"Successfully optimized and saved model to {output_path}")
    except Exception as e:
        logger.error(f"Failed to optimize ONNX model: {e}")
        # Fallback to copy if optimization fails
        import shutil

        logger.warning("Falling back to simple file copy.")
        shutil.copyfile(input_path, output_path)
