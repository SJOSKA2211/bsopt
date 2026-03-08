import torch
import torch.nn as nn


class OptionPricingNN(nn.Module):
    """
    Feed-forward Neural Network for Option Pricing.
    Supports quantization and pruning.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dims: list[int] | None = None,
        num_classes: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            last_dim = h

        layers.append(nn.Linear(last_dim, num_classes))  # Price or Logit output
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def export_onnx(self, path: str, input_sample: torch.Tensor):
        """Export model to ONNX format for efficient inference."""
        self.eval()
        torch.onnx.export(
            self,
            input_sample,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def apply_quantization(self):
        """Apply static quantization to the model."""
        self.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self, inplace=True)
        # Note: In a real scenario, we need a calibration pass here before conversion.
        # For demonstration, we convert directly after preparation.
        torch.quantization.convert(self, inplace=True)
        return self

    def apply_pruning(self, amount: float = 0.2):
        """
        Apply global unstructured pruning to linear layers.
        OPTIMIZED: Reduces model size and inference latency on supported backends.
        """
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
            # Make pruning permanent
            for module, name in parameters_to_prune:
                prune.remove(module, name)
        return self
