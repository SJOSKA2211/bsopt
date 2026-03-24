from typing import Any, cast

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.math_kernel.base import BasePricingEngine
from src.math_kernel.models import BSParameters, OptionGreeks
from src.ml.architectures.neural_network import OptionPricingNN

logger = structlog.get_logger(__name__)

class NeuralPricingEngine(BasePricingEngine):  # type: ignore
    """
    Pricing Engine powered by a Neural Network (MLP).
    Leverages PyTorch for pricing and automatic differentiation for Greeks.
    """

    def __init__(self, model_path: str | None = None) -> None:
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
    ) -> None:
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

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

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
        Reuses the optimized vectorized path.
        """
        res = self.price_batch(
            np.array([params.spot]),
            np.array([params.strike]),
            np.array([params.maturity]),
            np.array([params.volatility]),
            np.array([params.rate]),
            np.array([params.dividend]),
            np.array([option_type]),
        )
        return float(res[0])

    def optimize_for_inference(
        self, onnx_path: str | None = None, prune_amount: float = 0.2
    ) -> "NeuralPricingEngine":
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

    def price_batch(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray,
        rates: np.ndarray,
        dividends: np.ndarray,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """
        High-Performance Vectorized Batch Pricing.
        """
        data = np.stack([spots, strikes, maturities, vols, rates, dividends], axis=1)
        input_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            call_prices = self.model(input_tensor).squeeze().cpu().numpy()

        # Handle Put-Call Parity Vectorized
        is_call = (option_types == "call") | (option_types == "CALL")

        if np.all(is_call):
            return cast(np.ndarray[Any, np.dtype[np.float64]], call_prices)

        put_prices = (
            call_prices
            - spots * np.exp(-dividends * maturities)
            + strikes * np.exp(-rates * maturities)
        )
        return cast(
            np.ndarray[Any, np.dtype[np.float64]],
            np.where(is_call, call_prices, np.maximum(put_prices, 0.0)),
        )

    def price_batch_greeks(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vols: np.ndarray,
        rates: np.ndarray,
        dividends: np.ndarray,
        option_types: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        High-Performance Vectorized Greeks using Autograd Batching.
        """
        data = np.stack([spots, strikes, maturities, vols, rates, dividends], axis=1)
        input_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        input_tensor.requires_grad_(True)

        prices = self.model(input_tensor)

        # Batch Gradients
        grads = torch.autograd.grad(
            prices, input_tensor, grad_outputs=torch.ones_like(prices), create_graph=True
        )[0]

        delta_c = grads[:, 0].detach().cpu().numpy()
        vega_c = grads[:, 3].detach().cpu().numpy()
        theta_c = -grads[:, 2].detach().cpu().numpy()
        rho_c = grads[:, 4].detach().cpu().numpy()

        # Second order (Gamma) - Approximated for batch speed or use Hessian if needed
        # For performance, we'll use a centered difference on the already computed gradients
        gamma_grads = torch.autograd.grad(
            grads[:, 0], input_tensor, grad_outputs=torch.ones_like(grads[:, 0]), retain_graph=False
        )[0]
        gamma = gamma_grads[:, 0].detach().cpu().numpy()

        is_call = (option_types == "call") | (option_types == "CALL")

        # Vectorized Put Greeks via Parity
        delta = np.where(is_call, delta_c, delta_c - np.exp(-dividends * maturities))
        vega = vega_c  # Vega is same for call/put
        theta = np.where(
            is_call,
            theta_c,
            theta_c
            + dividends * spots * np.exp(-dividends * maturities)
            - rates * strikes * np.exp(-rates * maturities),
        )
        rho = np.where(is_call, rho_c, rho_c - strikes * maturities * np.exp(-rates * maturities))

        return (
            cast(np.ndarray[Any, np.dtype[np.float64]], delta),
            cast(np.ndarray[Any, np.dtype[np.float64]], gamma),
            cast(np.ndarray[Any, np.dtype[np.float64]], theta),
            cast(np.ndarray[Any, np.dtype[np.float64]], vega),
            cast(np.ndarray[Any, np.dtype[np.float64]], rho),
        )

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Calculate Greeks for a single set of parameters.
        Reuses the optimized vectorized path for consistency.
        """
        res = self.price_batch_greeks(
            np.array([params.spot], dtype=np.float64),
            np.array([params.strike], dtype=np.float64),
            np.array([params.maturity], dtype=np.float64),
            np.array([params.volatility], dtype=np.float64),
            np.array([params.rate], dtype=np.float64),
            np.array([params.dividend], dtype=np.float64),
            np.array([option_type]),
        )
        return OptionGreeks(
            delta=float(res[0][0]),
            gamma=float(res[1][0]),
            theta=float(res[2][0]),
            vega=float(res[3][0]),
            rho=float(res[4][0]),
        )
