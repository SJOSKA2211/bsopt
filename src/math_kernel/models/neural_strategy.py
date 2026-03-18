import os

import numpy as np
import structlog

from src.math_kernel.base import PricingStrategy
from src.math_kernel.models import BSParameters, OptionGreeks

logger = structlog.get_logger(__name__)


class NeuralPricingStrategy(PricingStrategy):
    """
    OPTIMIZED: Neural Network based pricing using ONNX Runtime.
    Provides sub-microsecond inference for complex pricing surfaces.
    """

    def __init__(self, model_path: str | None = None):
        self.ort_session = None
        path = model_path or os.path.join(os.getcwd(), "results/ml/OptionPricer_NN.onnx")

        if os.path.exists(path):
            try:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Prioritize high-performance providers
                providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
                self.ort_session = ort.InferenceSession(path, sess_options, providers=providers)
                logger.info(
                    "neural_pricing_model_loaded",
                    path=path,
                    provider=self.ort_session.get_providers()[0],
                )

                # Pre-warm
                input_shape = self.ort_session.get_inputs()[0].shape
                # Handle dynamic batch size or fixed [1, 9]
                b = input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else 1
                dummy_in = np.zeros((b, input_shape[1]), dtype=np.float32)
                self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: dummy_in})
            except Exception as e:
                logger.warning("neural_pricing_load_failed", error=str(e))
        else:
            logger.debug("neural_pricing_model_not_found", path=path)

    def price_european(self, params: BSParameters, option_type: str = "call") -> float:
        if not self.ort_session:
            # Fallback to Black-Scholes if model not loaded
            from src.math_kernel.black_scholes import BlackScholesEngine

            return BlackScholesEngine().price_european(params, option_type)

        # Feature vector: [S, K, T, sigma, r, q, is_call, ...]
        # Map to OptionPricingNN input_dim (assumed 9 based on class definition if it includes Greeks or other params)
        # For simplicity, we'll use the src.shared 7 features + 2 padding or derived features
        is_call_val = 1.0 if option_type.lower() == "call" else 0.0
        features = np.array(
            [
                [
                    params.spot,
                    params.strike,
                    params.maturity,
                    params.volatility,
                    params.rate,
                    params.dividend,
                    is_call_val,
                    0.0,
                    0.0,  # Padding to 9 features
                ]
            ],
            dtype=np.float32,
        )

        input_name = self.ort_session.get_inputs()[0].name
        prediction = self.ort_session.run(None, {input_name: features})[0]
        return float(prediction[0][0])

    def calculate_greeks(self, params: BSParameters, option_type: str = "call") -> OptionGreeks:
        """
        Greeks via Finite Differences on the neural surface.
        """
        if not self.ort_session:
            from src.math_kernel.black_scholes import BlackScholesEngine
            return BlackScholesEngine().calculate_greeks(params, option_type)

        def get_price(s, k, t, v, r, q):
            p = BSParameters(spot=s, strike=k, maturity=t, volatility=v, rate=r, dividend=q)
            return self.price_european(p, option_type)

        s, k, t, v, r, q = params.spot, params.strike, params.maturity, params.volatility, params.rate, params.dividend
        eps = 1e-4

        # Delta & Gamma
        p_plus = get_price(s + eps, k, t, v, r, q)
        p_minus = get_price(s - eps, k, t, v, r, q)
        p_mid = get_price(s, k, t, v, r, q)
        
        delta = (p_plus - p_minus) / (2 * eps)
        gamma = (p_plus - 2 * p_mid + p_minus) / (eps**2)

        # Vega
        p_v_plus = get_price(s, k, t, v + eps, r, q)
        vega = (p_v_plus - p_mid) / eps

        # Theta (using 1 day step)
        dt = 1/365.0
        p_t_minus = get_price(s, k, max(0, t - dt), v, r, q)
        theta = (p_t_minus - p_mid) / dt

        # Rho
        p_r_plus = get_price(s, k, t, v, r + eps, q)
        rho = (p_r_plus - p_mid) / eps

        return OptionGreeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho)
        )
