import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import structlog

logger = structlog.get_logger(__name__)

def quantize_onnx_model(model_path: str, output_path: str):
    """
    Quantize ONNX model to INT8 for ultra-low latency inference.
    """
    try:
        logger.info("quantizing_onnx_model", path=model_path)
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )
        logger.info("onnx_quantization_complete", output=output_path)
    except Exception as e:
        logger.error("onnx_quantization_failed", error=str(e))
        raise
