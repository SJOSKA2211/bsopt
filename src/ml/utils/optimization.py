import os
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)

def export_to_onnx(model: Any, dummy_input: torch.Tensor, export_path: str, input_names: list = None, output_names: list = None):
    """
    Optimizes a model for production by exporting it to ONNX format.
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    logger.info("exporting_to_onnx", path=export_path)
    
    try:
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Export logic for PyTorch
        if isinstance(model, torch.nn.Module):
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    input_names[0]: {0: 'batch_size'},
                    output_names[0]: {0: 'batch_size'}
                }
            )
            logger.info("onnx_export_success", path=export_path)
        else:
            logger.error("unsupported_model_type", type=type(model))
            
    except Exception as e:
        logger.error("onnx_export_failed", error=str(e))
        raise