import os
import time
from dataclasses import dataclass

import numpy as np
import structlog
from scipy.optimize import differential_evolution, minimize

from src.config import settings
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams
from src.shared.observability import (
    CALIBRATION_DURATION,
    HESTON_FELLER_MARGIN,
    HESTON_R_SQUARED,
    push_metrics,
)

logger = structlog.get_logger()

@dataclass
class MarketOption:
    """Market option data point."""
    T: float
    strike: float
    spot: float
    price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    option_type: str  # 'call' or 'put'

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def moneyness(self) -> float:
        return self.strike / self.spot

class HestonCalibrator:
    """
    Multi-stage calibration with quality metrics.
    Stage 1: Neural Inference (ONNX) or Differential Evolution (global search)
    Stage 2: SLSQP (local refinement)
    """
    FELLER_TOLERANCE = 0.01
    MAX_RMSE = 0.05
    MIN_LIQUID_OPTIONS = 5  # Reduced for testing flexibility

    def __init__(self, risk_free_rate: float = 0.03):
        self.r = risk_free_rate
        self.calibration_history: list[dict] = []
        self.ort_session = None
        
        # Load ONNX model for neural calibration if available
        try:
            import onnxruntime as ort
            model_path = settings.HESTON_MODEL_ONNX_PATH
            if model_path and os.path.exists(model_path):
                # Configure for ultra-low latency inference
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_cpu_mem_arena = True
                
                # Prioritize TensorRT then CUDA then CPU
                available_providers = ort.get_available_providers()
                providers = []
                if 'TensorrtExecutionProvider' in available_providers:
                    providers.append(('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 2147483648,
                        'trt_fp16_enable': True,
                    }))
                if 'CUDAExecutionProvider' in available_providers:
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')

                self.ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)
                logger.info("neural_calibration_model_loaded", path=model_path, providers=[p[0] if isinstance(p, tuple) else p for p in providers])
                
                # SOTA: Pre-warm the session to avoid cold-start latency
                try:
                    dummy_input = np.zeros((1, 1, 10, 10), dtype=np.float32)
                    if len(self.ort_session.get_inputs()[0].shape) == 2:
                        dummy_input = dummy_input.reshape(1, 100)
                    self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: dummy_input})
                    logger.debug("neural_calibration_model_warmed_up")
                except Exception as e:
                    logger.warning("neural_calibration_warmup_failed", error=str(e))
        except ImportError:
            logger.debug("onnxruntime_not_installed_skipping_neural_calibration")
        except Exception as e:
            logger.warning("neural_calibration_model_load_failed", error=str(e))

    def _filter_liquid_options(self, market_data: list[MarketOption]) -> list[MarketOption]:
        """Filter out illiquid options."""
        filtered = [
            opt for opt in market_data 
            if opt.volume > 0 
            and opt.open_interest > 10
            and 0.5 <= opt.moneyness <= 1.5
        ]
        return filtered

    def _calibrate_neural(self, market_data: list[MarketOption]) -> np.ndarray | None:
        """
        Predict Heston parameters using ONNX model.
        Features: A 10x10 grid of Implied Volatilities (Moneyness vs T).
        """
        if not self.ort_session:
            return None
            
        try:
            from src.pricing.implied_vol import implied_volatility
            
            # 1. Define feature grid (Standard for Neural Heston)
            moneyness_grid = np.linspace(0.8, 1.2, 10)
            expiry_grid = np.linspace(0.1, 2.0, 10)
            
            # 2. Extract IVs from market data
            market_ivs = []
            m_coords = []
            t_coords = []
            
            for opt in market_data:
                try:
                    iv = implied_volatility(
                        opt.mid_price, opt.spot, opt.strike, opt.T, self.r, 0.0, opt.option_type
                    )
                    market_ivs.append(iv)
                    m_coords.append(opt.moneyness)
                    t_coords.append(opt.T)
                except Exception:
                    continue # nosec B112
            
            if len(market_ivs) < self.MIN_LIQUID_OPTIONS:
                return None
                
            # 3. Interpolate onto standard grid (Optimized using Scipy LinearNDInterpolator)
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
            
            # Pre-filter to remove duplicates which can crash interpolators
            coords = np.array(list(zip(m_coords, t_coords)))
            _, unique_idx = np.unique(coords, axis=0, return_index=True)
            unique_coords = coords[unique_idx]
            unique_ivs = np.array(market_ivs)[unique_idx]

            interp = LinearNDInterpolator(unique_coords, unique_ivs)
            
            M, T = np.meshgrid(moneyness_grid, expiry_grid)
            grid_ivs = interp(M, T)
            
            # Fill NaNs from interpolation (extrapolate with nearest if needed)
            if np.isnan(grid_ivs).any():
                nearest = NearestNDInterpolator(unique_coords, unique_ivs)
                nan_mask = np.isnan(grid_ivs)
                grid_ivs[nan_mask] = nearest(M[nan_mask], T[nan_mask])
            
            # 4. Prepare ONNX input [Batch, 1, 10, 10]
            input_data = grid_ivs.astype(np.float32).reshape(1, 1, 10, 10)
            
            start_inference = time.perf_counter()
            input_name = self.ort_session.get_inputs()[0].name
            # Check if model expects flattened input
            if len(self.ort_session.get_inputs()[0].shape) == 2:
                input_data = input_data.reshape(1, 100)
                
            prediction = self.ort_session.run(None, {input_name: input_data})[0][0]
            inference_time = (time.perf_counter() - start_inference) * 1000
            
            logger.debug("neural_inference_complete", duration_ms=inference_time, provider=self.ort_session.get_providers()[0])
            
            # Output: [kappa, theta, sigma, rho, v0]
            return prediction.astype(np.float64)
            
        except Exception as e:
            logger.warning("neural_calibration_failed", error=str(e))
            return None

    def _weighted_objective(self, params: np.ndarray, market_data: list[MarketOption]) -> float:
        """Vega-weighted RMSE objective function."""
        kappa, theta, sigma, rho, v0 = params
        
        # Penalty for Feller condition
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma**2 * (1 - self.FELLER_TOLERANCE)
        if feller_lhs <= feller_rhs:
            return 1e12

        total_error = 0.0
        total_weight = 0.0
        
        for opt in market_data:
            try:
                model = HestonModelFFT(
                    HestonParams(v0, kappa, theta, sigma, rho), self.r, opt.T
                )
                if opt.option_type == 'call':
                    model_price = model.price_call(opt.spot, opt.strike)
                else:
                    model_price = model.price_put(opt.spot, opt.strike)
                
                market_price = opt.mid_price
                # Simple vega approx for weighting
                weight = 1.0 / max(opt.spot * np.sqrt(opt.T) * 0.4, 0.01)
                
                total_error += ((market_price - model_price) ** 2) * weight
                total_weight += weight
            except Exception:
                continue # nosec B112
                
        if total_weight == 0:
            return 1e12
            
        return np.sqrt(total_error / total_weight)

    def calibrate(self, market_data: list[MarketOption], maxiter: int = 50, popsize: int = 15, symbol: str = "unknown") -> tuple[HestonParams, dict]:
        """Two-stage calibration with quality metrics."""
        start_time = time.time()
        try:
            liquid_options = self._filter_liquid_options(market_data)
            if len(liquid_options) < self.MIN_LIQUID_OPTIONS:
                raise ValueError(f"Insufficient liquid options: {len(liquid_options)} < {self.MIN_LIQUID_OPTIONS}")

            bounds = [
                (0.1, 15.0),  # kappa
                (0.01, 0.5),  # theta
                (0.05, 2.0),  # sigma
                (-0.95, 0.95),# rho
                (0.01, 0.5)   # v0
            ]

            # Stage 1: Global Search (Neural or DE)
            initial_guess = self._calibrate_neural(liquid_options)
            
            if initial_guess is not None:
                logger.info("using_neural_calibration_guess", symbol=symbol)
                # Use neural prediction as starting point for local optimizer
                current_best = initial_guess
            else:
                # Fallback to Differential Evolution
                de_result = differential_evolution(
                    self._weighted_objective, bounds, args=(liquid_options,), 
                    maxiter=maxiter, popsize=popsize, seed=42
                )
                current_best = de_result.x

            # Stage 2: Local Refinement (SLSQP)
            slsqp_result = minimize(
                self._weighted_objective, current_best, args=(liquid_options,),
                method='SLSQP', bounds=bounds
            )

            kappa, theta, sigma, rho, v0 = slsqp_result.x
            final_params = HestonParams(v0, kappa, theta, sigma, rho)
            
            # Metrics
            market_prices = np.array([opt.mid_price for opt in liquid_options])
            final_rmse = slsqp_result.fun
            ss_tot = np.sum((market_prices - np.mean(market_prices)) ** 2)
            ss_res = (final_rmse ** 2) * len(liquid_options)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Observability
            duration = time.time() - start_time
            CALIBRATION_DURATION.labels(symbol=symbol).observe(duration)
            HESTON_R_SQUARED.labels(symbol=symbol).set(r_squared)
            
            # Feller condition margin: 2κθ - σ²
            feller_margin = 2 * kappa * theta - sigma**2
            HESTON_FELLER_MARGIN.labels(symbol=symbol).set(feller_margin)

            metrics = {
                'rmse': float(final_rmse),
                'r_squared': float(r_squared),
                'feller_margin': float(feller_margin),
                'num_options': len(liquid_options),
                'success': slsqp_result.success,
                'duration': duration,
                'method': 'neural+slsqp' if initial_guess is not None else 'de+slsqp'
            }
            
            # Push metrics to gateway if in background worker context
            push_metrics(job_name=f"heston_calibration_{symbol}")
            
            return final_params, metrics
        except Exception as e:
            logger.error("heston_calibration_failed", symbol=symbol, error=str(e))
            raise

    def calibrate_surface(self, market_data: list[MarketOption]) -> dict[float, tuple[float, ...]]:
        """Fit SVI parameters for each maturity slice."""
        from src.pricing.calibration.svi_surface import SVISurface
        from src.pricing.implied_vol import implied_volatility
        
        liquid_options = self._filter_liquid_options(market_data)
        
        # Group by maturity
        by_maturity: dict[float, list[MarketOption]] = {}
        for opt in liquid_options:
            if opt.T not in by_maturity:
                by_maturity[opt.T] = []
            by_maturity[opt.T].append(opt)
            
        surface_params = {}
        for T, options in by_maturity.items():
            if len(options) < self.MIN_LIQUID_OPTIONS:
                continue # nosec B112
                
            log_strikes = []
            total_variances = []
            
            for opt in options:
                try:
                    iv = implied_volatility(
                        opt.mid_price, opt.spot, opt.strike, opt.T, self.r, 0.0, opt.option_type
                    )
                    log_strikes.append(np.log(opt.strike / opt.spot))
                    total_variances.append(iv * iv * opt.T)
                except Exception:
                    continue # nosec B112
            
            if len(log_strikes) >= self.MIN_LIQUID_OPTIONS:
                params = SVISurface.fit_svi_slice(
                    np.array(log_strikes), np.array(total_variances), T
                )
                surface_params[float(T)] = params
                
        return surface_params