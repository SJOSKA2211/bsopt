import os
from typing import Any, Final, cast

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

# Constants for configuration-driven design
DEFAULT_INPUT_DIM: Final[int] = int(os.getenv("ML_INPUT_DIM", "9"))
DEFAULT_HIDDEN_DIMS: Final[list[int]] = [int(d) for d in os.getenv("ML_HIDDEN_DIMS", "128,128,64").split(",")]
DEFAULT_BATCH_SIZE: Final[int] = int(os.getenv("ML_BATCH_SIZE", "32"))
DEFAULT_LEARNING_RATE: Final[float] = float(os.getenv("ML_LEARNING_RATE", "0.001"))

class NeuralPricingEngine(BasePricingEngine):  # type: ignore
    """
    Pricing Engine powered by a Neural Network (MLP).
    Leverages PyTorch for pricing and automatic differentiation for Greeks.
    Execution is strictly CPU-bound for lightweight deployment.
    """

    def __init__(self, model_path: str | None = None) -> None:
        # Force CPU-only execution as per system mandate
        self.device = torch.device("cpu")
        
        self.model = OptionPricingNN(
            input_dim=DEFAULT_INPUT_DIM, 
            hidden_dims=DEFAULT_HIDDEN_DIMS, 
            num_classes=1
        ).to(self.device)

        if not model_path:
            model_path = os.getenv("ML_MODEL_PATH")

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                logger.info("neural_engine_model_loaded", path=model_path)
            except Exception as e:
                logger.error("neural_engine_load_failed", error=str(e))
        
        self.model.eval()

    def train_model(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        """
        Train the underlying neural network.
        Inputs expected shape (N, input_dim).
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
        """Convert BSParameters to a standardized feature tensor with gradients enabled."""
        if params.strike == 0:
             raise ValueError("strike_cannot_be_zero")
             
        moneyness = params.spot / params.strike
        log_moneyness = np.log(moneyness)
        sqrt_t = np.sqrt(params.maturity)
        days_to_t = params.maturity * 365.0

        # Feature vector is configuration-driven based on input_dim
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
        
        # Slice or pad data based on expected input_dim if needed, but here we assume dim=9
        tensor = torch.tensor([data[:DEFAULT_INPUT_DIM]], dtype=torch.float32, device=self.device)
        tensor.requires_grad_(True)
        return tensor

    def price(self, params: BSParameters, option_type: str = "call") -> float:
        """Calculate option price using the Neural Network."""
        is_call = option_type.lower() == "call"
        input_tensor = self._params_to_tensor(params, is_call)

        with torch.no_grad():
            price_output = self.model(input_tensor)
            return float(price_output.item())

    def optimize_for_inference(
        self, onnx_path: str | None = None, prune_amount: float = 0.2
    ) -> "NeuralPricingEngine":
        """Optimize model for inference using pruning and ONNX export."""
        self.model.apply_pruning(amount=prune_amount)

        if not onnx_path:
            onnx_path = os.getenv("ML_ONNX_EXPORT_PATH")

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
        """Vectorized Batch Pricing using standardized feature vector."""
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

        input_tensor = torch.tensor(data[:, :DEFAULT_INPUT_DIM], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prices = self.model(input_tensor).squeeze().numpy()

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized Greeks using Autograd Batching."""
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

        input_tensor = torch.tensor(data[:, :DEFAULT_INPUT_DIM], dtype=torch.float32, device=self.device)
        input_tensor.requires_grad_(True)

        prices = self.model(input_tensor)

        grads = torch.autograd.grad(
            prices, input_tensor, grad_outputs=torch.ones_like(prices), create_graph=True
        )[0]

        delta = grads[:, 0].detach().numpy()
        theta = -grads[:, 2].detach().numpy()
        vega = grads[:, 8].detach().numpy()

        gamma_grads = torch.autograd.grad(
            grads[:, 0], input_tensor, grad_outputs=torch.ones_like(grads[:, 0]), retain_graph=False
        )[0]
        gamma = gamma_grads[:, 0].detach().numpy()

        rho = np.zeros_like(delta)

        return (
            cast(np.ndarray[Any, np.dtype[np.float64]], delta),
            cast(np.ndarray[Any, np.dtype[np.float64]], gamma),
            cast(np.ndarray[Any, np.dtype[np.float64]], theta),
            cast(np.ndarray[Any, np.dtype[np.float64]], vega),
            cast(np.ndarray[Any, np.dtype[np.float64]], rho),
        )

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """Calculate Greeks for a single set of parameters."""
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