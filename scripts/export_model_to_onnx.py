import torch
import torch.onnx

from src.ml.architectures.neural_network import OptionPricingNN


def export_to_onnx(model_path: str, output_path: str, input_dim: int):
    """
    Export a PyTorch OptionPricingNN model to ONNX format.
    """
    # Load the model
    # Assuming hidden_dims are saved in the state_dict or we know them
    # For simplicity, we use the default [128, 64, 32]
    model = OptionPricingNN(input_dim=input_dim, hidden_dims=[128, 64, 32])

    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, input_dim)

    # Export
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
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python export_model_to_onnx.py <model_path> <output_path> <input_dim>"
        )
    else:
        export_to_onnx(sys.argv[1], sys.argv[2], int(sys.argv[3]))
