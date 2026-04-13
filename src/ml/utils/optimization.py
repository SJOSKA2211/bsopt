import os
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


def export_to_onnx(
    model: Any,
    dummy_input: torch.Tensor,
    export_path: str,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> None:
    """
    Optimizes a model for production by exporting it to ONNX format.
    """
    actual_input_names = input_names if input_names is not None else ["input"]
    actual_output_names = output_names if output_names is not None else ["output"]

    logger.info("exporting_to_onnx", path=export_path)

    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Export logic for PyTorch
        if isinstance(model, torch.nn.Module):
            torch.onnx.export(
                model,
                (dummy_input,),
                export_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=actual_input_names,
                output_names=actual_output_names,
                dynamic_axes={
                    actual_input_names[0]: {0: "batch_size"},
                    actual_output_names[0]: {0: "batch_size"},
                },
            )
            logger.info("onnx_export_success", path=export_path)
        else:
            logger.error("unsupported_model_type", type=type(model))

    except Exception as e:
        logger.error("onnx_export_failed", error=str(e))
        raise


def quantize_onnx_model(model_path: str, output_path: str) -> None:
    """
    Apply INT8 quantization to an ONNX model.
    OPTIMIZED: Reduces model size by ~4x and improves CPU throughput.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        logger.info("quantizing_onnx_model", input=model_path, output=output_path)
        quantize_dynamic(
            model_input=model_path, model_output=output_path, weight_type=QuantType.QUInt8
        )
        logger.info("quantization_success", path=output_path)
    except Exception as e:
        logger.error("quantization_failed", error=str(e))
        raise