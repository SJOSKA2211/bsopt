import torch
import torch.nn as nn
import structlog
from typing import Any

logger = structlog.get_logger()

class ModelQuantizer:
    """
    SOTA Model Quantization Strategy.
    Reduces model size and increases inference speed using PyTorch quantization.
    """
    def __init__(self):
        logger.info("model_quantizer_initialized")

    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Applies dynamic quantization to the provided model.
        Useful for models where the weights are static but activations vary.
        """
        logger.info("applying_dynamic_quantization", model_type=type(model).__name__)
        
        try:
            # We target Linear and LSTM layers for quantization as they are compute-heavy
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM}, 
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            logger.error("quantization_failed", error=str(e))
            return model

    def save_quantized_model(self, model: nn.Module, path: str):
        """
        Saves the quantized model state dict.
        """
        torch.save(model.state_dict(), path)
        logger.info("quantized_model_saved", path=path)
