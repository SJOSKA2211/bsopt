import pytest
import torch
import torch.nn as nn
import os
from src.ml.serving.quantization import ModelQuantizer
from unittest.mock import patch, MagicMock

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

def test_dynamic_quantization():
    model = SimpleNet()
    quantizer = ModelQuantizer()
    quantized_model = quantizer.quantize_dynamic(model)
    assert hasattr(quantized_model.fc, "weight")
    assert quantized_model.fc.weight().dtype == torch.qint8

def test_quantization_failure():
    model = SimpleNet()
    quantizer = ModelQuantizer()
    with patch("torch.quantization.quantize_dynamic", side_effect=Exception("Quant fail")):
        # Should return original model on failure
        res = quantizer.quantize_dynamic(model)
        assert res == model

def test_quantization_save_load(tmp_path):
    model = SimpleNet()
    quantizer = ModelQuantizer()
    quantized_model = quantizer.quantize_dynamic(model)
    
    save_path = str(tmp_path / "quantized_model.pth")
    quantizer.save_quantized_model(quantized_model, save_path)
    assert os.path.exists(save_path)