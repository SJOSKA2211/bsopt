import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.ml.architectures.neural_network import OptionPricingNN
from src.pricing.base import BasePricingEngine
from src.pricing.models import BSParameters, OptionGreeks


class NeuralPricingEngine(BasePricingEngine):
    """
    Pricing Engine powered by a Neural Network (MLP).
    Leverages PyTorch for pricing and automatic differentiation for Greeks.
    """

    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Standard BS Inputs: Spot, Strike, Maturity, Volatility, Rate, Dividend (6)
        self.model = OptionPricingNN(input_dim=6, hidden_dims=[128, 128, 64], num_classes=1).to(
            self.device
        )

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # nosec B614

        self.model.eval()

    def train_model(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        """
        Train the underlying neural network.

        Args:
            inputs: (N, 6) numpy array of BS parameters.
            targets: (N, 1) numpy array of option prices.
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.
        """
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        x_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Logging could go here, but keeping it clean.
            pass

        self.model.eval()

    def _params_to_tensor(self, params: BSParameters) -> torch.Tensor:
        """Convert BSParameters to a tensor with gradients enabled."""
        data = [
            params.spot,
            params.strike,
            params.maturity,
            params.volatility,
            params.rate,
            params.dividend,
        ]
        tensor = torch.tensor([data], dtype=torch.float32, device=self.device)
        tensor.requires_grad_(True)
        return tensor

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """
        Calculate option price using the Neural Network.
        Note: Currently only supports 'call'. Put-Call parity should be handled by caller or wrapper.
        """
        if option_type.lower() != "call":
            # For now, simplistic implementation. Real version would handle puts via parity or separate output.
            raise NotImplementedError("NeuralEngine currently only supports Call options directly.")

        input_tensor = self._params_to_tensor(params)
        with torch.no_grad():
            prediction = self.model(input_tensor)

        return prediction.item()

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Calculate Greeks using PyTorch Autograd.
        This provides exact derivatives of the model's pricing function.
        """
        if option_type.lower() != "call":
            raise NotImplementedError("NeuralEngine currently only supports Call options directly.")

        input_tensor = self._params_to_tensor(params)

        # Forward pass
        price = self.model(input_tensor)

        # Backward pass (compute gradients w.r.t inputs)
        # Inputs: [Spot (0), Strike (1), T (2), Sigma (3), R (4), Q (5)]
        grads = torch.autograd.grad(price, input_tensor, create_graph=True)[0][0]

        delta = grads[0].item()  # dPrice/dSpot
        vega = grads[3].item()  # dPrice/dVol
        theta = -grads[
            2
        ].item()  # dPrice/dTime (Time to maturity decreases, so theta is usually negative derivative w.r.t T)
        rho = grads[4].item()  # dPrice/dRate

        # Second order: Gamma (d2Price/dSpot2)
        # We need to retain graph or run another backward pass on delta
        # Since we just need one element, let's use the delta we just computed?
        # No, grads[0] is a tensor attached to the graph.

        gamma_grad = torch.autograd.grad(grads[0], input_tensor, retain_graph=False)[0][0]
        gamma = gamma_grad[0].item()

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
