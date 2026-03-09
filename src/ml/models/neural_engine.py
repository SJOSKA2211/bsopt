import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.ml.architectures.neural_network import OptionPricingNN
from src.pricing.base import BasePricingEngine
from src.pricing.models import BSParameters, OptionGreeks

logger = structlog.get_logger(__name__)


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
        OPTIMIZED: Uses Put-Call Parity to support Puts without separate model.
        """
        input_tensor = self._params_to_tensor(params)
        with torch.no_grad():
            call_price = self.model(input_tensor).item()

        if option_type.lower() == "call":
            return call_price
        else:
            # Put-Call Parity: P = C - S + K * exp(-r * T)
            # Spot (0), Strike (1), T (2), Sigma (3), R (4), Q (5)
            s = params.spot
            k = params.strike
            t = params.maturity
            r = params.rate
            q = params.dividend
            put_price = call_price - s * np.exp(-q * t) + k * np.exp(-r * t)
            return max(float(put_price), 0.0)

    def optimize_for_inference(self, onnx_path: str | None = None, prune_amount: float = 0.2):
        """
        Fine-tune model for zero-latency inference.
        """
        self.model.apply_pruning(amount=prune_amount)

        if onnx_path:
            sample = self._params_to_tensor(
                BSParameters(
                    spot=100, strike=100, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.01
                )
            )
            self.model.export_onnx(onnx_path, sample)
            logger.info("model_optimized_onnx", path=onnx_path)

        return self

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Calculate Greeks using PyTorch Autograd.
        This provides exact derivatives of the model's pricing function.
        OPTIMIZED: Uses Put-Call Parity relations for Put Greeks.
        """
        input_tensor = self._params_to_tensor(params)

        # Forward pass (always Call price)
        price = self.model(input_tensor)

        # Backward pass (compute gradients w.r.t inputs)
        # Inputs: [Spot (0), Strike (1), T (2), Sigma (3), R (4), Q (5)]
        grads = torch.autograd.grad(price, input_tensor, create_graph=True)[0][0]

        call_delta = grads[0].item()  # dPrice/dSpot
        call_vega = grads[3].item()  # dPrice/dVol
        call_theta = -grads[2].item()  # dPrice/dTime
        call_rho = grads[4].item()  # dPrice/dRate

        # Second order: Gamma (same for Call/Put)
        gamma_grad = torch.autograd.grad(grads[0], input_tensor, retain_graph=False)[0][0]
        gamma = gamma_grad[0].item()

        if option_type.lower() == "call":
            return OptionGreeks(
                delta=call_delta, gamma=gamma, theta=call_theta, vega=call_vega, rho=call_rho
            )
        else:
            # Put Greeks via Parity (assuming q=dividend, r=rate)
            t = params.maturity
            r = params.rate
            q = params.dividend
            k = params.strike

            # Delta_p = Delta_c - exp(-qT)
            put_delta = call_delta - np.exp(-q * t)
            # Vega_p = Vega_c
            # Gamma_p = Gamma_c
            # Theta_p = Theta_c + q*S*exp(-qT) - r*K*exp(-rT)
            put_theta = call_theta + q * params.spot * np.exp(-q * t) - r * k * np.exp(-r * t)
            # Rho_p = Rho_c - K*T*exp(-rT)
            put_rho = call_rho - k * t * np.exp(-r * t)

            return OptionGreeks(
                delta=put_delta, gamma=gamma, theta=put_theta, vega=call_vega, rho=put_rho
            )
