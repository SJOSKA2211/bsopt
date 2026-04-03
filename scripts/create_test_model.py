import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class OptionPricingNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[128, 64], output_dim=1):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_and_export_model():
    # Matches the 9 features expected by serve.py
    input_dim = 9
    model = OptionPricingNN(input_dim=input_dim)
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_dim)

    # Export to ONNX
    output_path = "models/latest_pricing.onnx"
    Path("models").mkdir(exist_ok=True)
    
    print(f"[*] Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[+] Model exported successfully. Size: {Path(output_path).stat().st_size} bytes")

if __name__ == "__main__":
    create_and_export_model()
