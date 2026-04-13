from typing import cast

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()


class ModelQuantizer:
    """
    OPTIMIZED Model Quantization Strategy.
    Reduces model size and increases inference speed using PyTorch quantization.
    """

    def __init__(self) -> None:
        logger.info("model_quantizer_initialized")

    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Applies dynamic quantization to the provided model.
        Useful for models where the weights are static but activations vary.
        """
        logger.info("applying_dynamic_quantization", model_type=type(model).__name__)

        try:
            # We target Linear and LSTM layers for quantization as they are compute-heavy
            # mypy might struggle with torch.quantization if stubs are missing, casting to nn.Module
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
            )
            return cast(nn.Module, quantized_model)
        except Exception as e:
            logger.error("quantization_failed", error=str(e))
            return model

    def save_quantized_model(self, model: nn.Module, path: str) -> None:
        """
        Saves the quantized model state dict.
        """
        torch.save(model.state_dict(), path)
        logger.info("quantized_model_saved", path=path)

    def quantize_onnx_model(self, input_path: str, output_path: str) -> None:
        """
        Performs INT8 quantization on an ONNX model for high-performance inference.
        """
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

            logger.info("quantizing_onnx_model", input=input_path, output=output_path)
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
            )
            logger.info("onnx_quantization_success")
        except Exception as e:
            logger.error("onnx_quantization_failed", error=str(e))
            raise