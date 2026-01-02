import logging
import time
from typing import cast

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """
    High-performance inference engine using ONNX Runtime.
    """

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

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
    This is a placeholder implementation that copies the model.
    Actual optimization requires specialized ONNX tools (e.g., onnxoptimizer)
    or ONNX Runtime's optimization capabilities during session creation.
    """
    logger.warning(f"ONNX model optimization is a placeholder. Copying {input_path} to {output_path}.")
    import shutil
    try:
        shutil.copyfile(input_path, output_path)
    except Exception as e:
        logger.error(f"Failed to copy ONNX model during placeholder optimization: {e}")
