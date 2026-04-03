"""
ONNX Quantization Pipeline for DT-v2 Models

Applies INT8 dynamic quantization to ONNX models to achieve sub-millisecond
inference performance on CPU-bound edge nodes.
"""

import os

import structlog
from onnxruntime.quantization import QuantType, quantize_dynamic

logger = structlog.get_logger(__name__)


class ONNXQuantizer:
    """
    Handles ONNX export and quantization for high-performance inference.
    """

    @staticmethod
    def quantize(input_onnx_path: str, output_onnx_path: str):
        """
        Apply INT8 dynamic quantization to an ONNX model.
        """
        logger.info("onnx_quantization_start", input=input_onnx_path)

        if not os.path.exists(input_onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {input_onnx_path}")

        try:
            quantize_dynamic(
                model_input=input_onnx_path,
                model_output=output_onnx_path,
                weight_type=QuantType.QUInt8,
                extra_options={
                    "ForceQuantizeNoInputCheck": True,
                    "MatMulConstBOnly": True,
                },
            )

            orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
            quant_size = os.path.getsize(output_onnx_path) / (1024 * 1024)

            logger.info(
                "onnx_quantization_complete",
                output=output_onnx_path,
                original_size_mb=round(orig_size, 2),
                quantized_size_mb=round(quant_size, 2),
                reduction=f"{round((1 - quant_size / orig_size) * 100, 1)}%",
            )
        except Exception as e:
            logger.error("onnx_quantization_failed", error=str(e))
            raise


if __name__ == "__main__":
    # Example integration test
    import sys

    if len(sys.argv) > 2:
        ONNXQuantizer.quantize(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python onnx_quantizer.py <input.onnx> <output_quant.onnx>")
