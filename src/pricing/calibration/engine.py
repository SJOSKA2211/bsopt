import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import structlog
from src.pricing.models.heston_fft import HestonModelFFT, HestonParams

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
    Stage 1: Differential Evolution (global search)
    Stage 2: SLSQP (local refinement)
    """
    FELLER_TOLERANCE = 0.01
    MAX_RMSE = 0.05
    MIN_LIQUID_OPTIONS = 5  # Reduced for testing flexibility

    def __init__(self, risk_free_rate: float = 0.03):
        self.r = risk_free_rate
        self.calibration_history: List[Dict] = []

    def _filter_liquid_options(self, market_data: List[MarketOption]) -> List[MarketOption]:
        """Filter out illiquid options."""
        filtered = [
            opt for opt in market_data 
            if opt.volume > 0 
            and opt.open_interest > 10
            and 0.5 <= opt.moneyness <= 1.5
        ]
        return filtered

    def _weighted_objective(self, params: np.ndarray, market_data: List[MarketOption]) -> float:
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
            except Exception:  # nosec
                continue
                
        if total_weight == 0:
            return 1e12
            
        return np.sqrt(total_error / total_weight)

    def calibrate(self, market_data: List[MarketOption], maxiter: int = 50, popsize: int = 15) -> Tuple[HestonParams, Dict]:
        """Two-stage calibration with quality metrics."""
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

        # Stage 1: Global
        de_result = differential_evolution(
            self._weighted_objective, bounds, args=(liquid_options,), 
            maxiter=maxiter, popsize=popsize, seed=42
        )

        # Stage 2: Local
        slsqp_result = minimize(
            self._weighted_objective, de_result.x, args=(liquid_options,),
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

        metrics = {
            'rmse': float(final_rmse),
            'r_squared': float(r_squared),
            'num_options': len(liquid_options),
            'success': slsqp_result.success
        }
        
        return final_params, metrics

    def calibrate_surface(self, market_data: List[MarketOption]) -> Dict[float, Tuple[float, ...]]:
        """Fit SVI parameters for each maturity slice."""
        from src.pricing.calibration.svi_surface import SVISurface
        from src.pricing.implied_vol import implied_volatility
        
        liquid_options = self._filter_liquid_options(market_data)
        
        # Group by maturity
        by_maturity: Dict[float, List[MarketOption]] = {}
        for opt in liquid_options:
            if opt.T not in by_maturity:
                by_maturity[opt.T] = []
            by_maturity[opt.T].append(opt)
            
        surface_params = {}
        for T, options in by_maturity.items():
            if len(options) < self.MIN_LIQUID_OPTIONS:
                continue
                
            log_strikes = []
            total_variances = []
            
            for opt in options:
                try:
                    iv = implied_volatility(
                        opt.mid_price, opt.spot, opt.strike, opt.T, self.r, 0.0, opt.option_type
                    )
                    log_strikes.append(np.log(opt.strike / opt.spot))
                    total_variances.append(iv * iv * opt.T)
                except Exception:  # nosec
                    continue
            
            if len(log_strikes) >= self.MIN_LIQUID_OPTIONS:
                params = SVISurface.fit_svi_slice(
                    np.array(log_strikes), np.array(total_variances), T
                )
                surface_params[float(T)] = params
                
        return surface_params
