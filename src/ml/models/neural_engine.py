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
    OPTIMIZED: Standardized 9-feature vector for high-fidelity pricing.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Standardized 9 Features: [Spot, Strike, T, is_call, moneyness, log_moneyness, sqrt_T, days_to_T, vol]
        self.model = OptionPricingNN(input_dim=9, hidden_dims=[128, 128, 64], num_classes=1).to(
            self.device
        )

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # nosec B614
                logger.info("neural_engine_model_loaded", path=model_path)
            except Exception as e:
                logger.error("neural_engine_load_failed", error=str(e))

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
        Inputs expected shape (N, 9).
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

    def _params_to_tensor(self, params: BSParameters, is_call: bool = True) -> torch.Tensor:
        """Convert BSParameters to a standardized 9-feature tensor with gradients enabled."""
        # Derived features
        moneyness = params.spot / params.strike
        log_moneyness = np.log(moneyness)
        sqrt_t = np.sqrt(params.maturity)
        days_to_t = params.maturity * 365.0

        data = [
            params.spot,
            params.strike,
            params.maturity,
            float(is_call),
            moneyness,
            log_moneyness,
            sqrt_t,
            days_to_t,
            params.volatility,
        ]
        tensor = torch.tensor([data], dtype=torch.float32, device=self.device)
        tensor.requires_grad_(True)
        return tensor

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """
        Calculate option price using the Neural Network.
        """
        is_call = option_type.lower() == "call"
        input_tensor = self._params_to_tensor(params, is_call)

        with torch.no_grad():
            price = self.model(input_tensor).item()

        return float(price)

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
        High-Performance Vectorized Batch Pricing using standardized 9-feature vector.
        """
        len(spots)
        is_call = (option_types == "call") | (option_types == "CALL")

        # Construct 9 features
        moneyness = spots / strikes
        log_moneyness = np.log(moneyness)
        sqrt_t = np.sqrt(maturities)
        days_to_t = maturities * 365.0

        data = np.stack(
            [
                spots,
                strikes,
                maturities,
                is_call.astype(np.float32),
                moneyness,
                log_moneyness,
                sqrt_t,
                days_to_t,
                vols,
            ],
            axis=1,
        )

        input_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prices = self.model(input_tensor).squeeze().cpu().numpy()

        return cast(np.ndarray[Any, np.dtype[np.float64]], prices)

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
        len(spots)
        is_call = (option_types == "call") | (option_types == "CALL")

        moneyness = spots / strikes
        log_moneyness = np.log(moneyness)
        sqrt_t = np.sqrt(maturities)
        days_to_t = maturities * 365.0

        data = np.stack(
            [
                spots,
                strikes,
                maturities,
                is_call.astype(np.float32),
                moneyness,
                log_moneyness,
                sqrt_t,
                days_to_t,
                vols,
            ],
            axis=1,
        )

        input_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
        input_tensor.requires_grad_(True)

        prices = self.model(input_tensor)

        # Batch Gradients
        grads = torch.autograd.grad(
            prices, input_tensor, grad_outputs=torch.ones_like(prices), create_graph=True
        )[0]

        delta = grads[:, 0].detach().cpu().numpy()
        theta = -grads[:, 2].detach().cpu().numpy()
        vega = grads[:, 8].detach().cpu().numpy()

        # Second order (Gamma)
        gamma_grads = torch.autograd.grad(
            grads[:, 0], input_tensor, grad_outputs=torch.ones_like(grads[:, 0]), retain_graph=False
        )[0]
        gamma = gamma_grads[:, 0].detach().cpu().numpy()

        # Rho (Approximate or use parity if rates were in features)
        # In this 9-feature set, rates/dividends are not features (standard for this specific engine)
        # If needed, parity can be applied if we assume Black-Scholes dynamics.
        rho = np.zeros_like(delta)

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
