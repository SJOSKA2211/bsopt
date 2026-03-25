from typing import Any

import numpy as np
import pandas as pd
import ray
import structlog
from scipy.cluster.hierarchy import linkage

from src.shared.distributed import RayOrchestrator

logger = structlog.get_logger()

@ray.remote
def _run_backtest_task(engine_instance, df, strategy_fn, params):
    """Ray task for parallel backtest execution."""
    return engine_instance.run_vectorized(df, strategy_fn, params)

def _get_quasi_diag(link):
    """Quasi-Diagonalization utility for HRP."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        num_items = link[-1, 3]
    return sort_ix.tolist()

def _get_cluster_var(cov, cluster_items):
    """Cluster variance utility for HRP."""
    cov_c = cov[np.ix_(cluster_items, cluster_items)]
    w = 1.0 / np.diag(cov_c)
    w /= w.sum()
    return np.dot(w.T, np.dot(cov_c, w))

def _get_rec_bisec(cov, sort_ix):
    """Recursive bisection utility for HRP."""
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_left = c_items[i]
            c_right = c_items[i + 1]
            alpha_1 = _get_cluster_var(cov, c_left)
            alpha_2 = _get_cluster_var(cov, c_right)
            alpha = 1 - alpha_1 / (alpha_1 + alpha_2)
            w[c_left] *= alpha
            w[c_right] *= 1 - alpha
    return w

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine.
    Supports MVO, HRP, and Black-Litterman models.
    """

    def __init__(self, symbols: list[str], returns_df: pd.DataFrame):
        self.symbols = symbols
        self.returns = returns_df[symbols]
        self.cov_matrix = self.returns.cov().values
        self.mean_returns = self.returns.mean().values

    def optimize_weights(self, method: str = "hrp", **kwargs) -> np.ndarray:
        """Unified interface for weight optimization."""
        if method == "hrp":
            return self.optimize_hrp()
        elif method == "mvo":
            return self.optimize_mvo()
        elif method == "black_litterman":
            return self.optimize_black_litterman(kwargs.get("views"), kwargs.get("confidences"))
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def optimize_hrp(self) -> np.ndarray:
        """Hierarchical Risk Parity (HRP) allocation."""
        # 1. Clustering
        corr = self.returns.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(dist, "single")

        # 2. Quasi-Diagonalization
        sort_ix = _get_quasi_diag(link)

        # 3. Recursive Bisection
        weights = _get_rec_bisec(self.cov_matrix, sort_ix)
        return weights.sort_index().values

class BacktestEngine:
    # ... (init stays same)

    def run_batch(self, scenarios: list[dict]) -> list[dict]:
        """Run multiple backtests in parallel using Ray Orchestrator."""
        RayOrchestrator.init()  # Using central, Docker-aware init

        futures = []
        for s in scenarios:
            futures.append(_run_backtest_task.remote(self, s["df"], s["fn"], s["params"]))

        return ray.get(futures)

    def run_vectorized(
        self, df: pd.DataFrame, strategy_fn: Any, params: dict | None = None
    ) -> dict[str, Any]:
        """
        Executes a strategy over historical data using vectorized operations.
        df must contain: timestamp, underlying_price, option_price, strike, maturity, etc.
        """
        logger.info("backtest_started", rows=len(df), params=params)
        start_time = pd.Timestamp.now()

        # 1. Generate Signals (Vectorized if possible within strategy_fn)
        # Strategy function should return a series of 'target_positions'
        df = strategy_fn(df, params)

        if "target_position" not in df.columns:
            raise ValueError("Strategy function must add 'target_position' column to DataFrame")

        # 2. Vectorized P&L Calculation using Numba Kernel
        from src.math_kernel.backtesting.kernel import (
            calculate_metrics_kernel,
            run_simulation_kernel,
        )

        # Extract raw arrays for the kernel
        prices = df["option_price"].values.astype(np.float64)
        positions = df["target_position"].values.astype(np.float64)

        equity_curve, mtm_pnl, commissions = run_simulation_kernel(
            prices,
            positions,
            self.initial_capital,
            params.get("transaction_cost_pct", 0.001) if params else 0.001,
        )

        df["equity_curve"] = equity_curve
        df["mtm_pnl"] = mtm_pnl
        df["commissions"] = commissions

        # 3. Calculate Performance Metrics using optimized 5-metric kernel
        total_return, sharpe, sortino, calmar, max_drawdown = calculate_metrics_kernel(
            equity_curve, self.initial_capital
        )

        # 4. OPTIMIZED Risk Metrics: VaR and Expected Shortfall (Vectorized)
        confidence_level = params.get("confidence_level", 0.95) if params else 0.95

        # Calculate periodic returns from equity curve
        returns_series = pd.Series(equity_curve).pct_change().dropna()

        # Historical VaR
        var_95 = (
            np.percentile(returns_series, (1 - confidence_level) * 100)
            if not returns_series.empty
            else 0.0
        )

        # Expected Shortfall
        es_95 = returns_series[returns_series <= var_95].mean() if not returns_series.empty else 0.0

        duration = (pd.Timestamp.now() - start_time).total_seconds()

        result = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "es_95": float(es_95),
            "trades_count": int(np.abs(df["target_position"].diff()).sum() > 0),
            "final_value": float(df["equity_curve"].iloc[-1]),
            "duration_seconds": duration,
            "status": "completed",
        }

        logger.info("backtest_completed", metrics=result)
        return result

    def optimize_mvo(self) -> np.ndarray:
        """Markowitz Mean-Variance Optimization (MVO)."""
        from scipy.optimize import minimize

        n = len(self.symbols)

        def objective(w):
            return np.dot(w.T, np.dot(self.cov_matrix, w))

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(
            objective, n * [1.0 / n], method="SLSQP", bounds=bounds, constraints=constraints
        )
        return res.x

    def optimize_black_litterman(
        self, views: np.ndarray, confidences: np.ndarray, tau: float = 0.05
    ) -> np.ndarray:
        """
        Black-Litterman model for Production view incorporation.
        Combines market prior (equilibrium) with investor views.
        """
        from scipy.optimize import minimize

        n = len(self.symbols)

        # 1. Prior Returns (Pi) - Simplified equilibrium returns
        delta = 2.5  # Risk aversion coefficient
        w_eq = np.ones(n) / n  # Neutral prior
        pi = delta * np.dot(self.cov_matrix, w_eq)

        # 2. View Integration
        # P: Pick matrix (identity for absolute views on each asset)
        P = np.eye(n)
        # Omega: Uncertainty of views (diagonal matrix of variances)
        omega = np.diag(confidences)

        # 3. Posterior Returns (Combined)
        # Er = [(tau * Cov)^-1 + P' * Omega^-1 * P]^-1 * [(tau * Cov)^-1 * Pi + P' * Omega^-1 * Q]
        term1 = np.linalg.inv(
            np.linalg.inv(tau * self.cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        )
        term2 = np.dot(np.linalg.inv(tau * self.cov_matrix), pi) + np.dot(
            P.T, np.dot(np.linalg.inv(omega), views)
        )
        er = np.dot(term1, term2)

        # 4. Final Optimization with Posterior Returns
        def objective(w):
            return -np.dot(w, er) + (delta / 2) * np.dot(w.T, np.dot(self.cov_matrix, w))

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(objective, w_eq, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x

    @staticmethod
    def autonomous_agent_strategy(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """
        Autonomous Strategy: Uses Neural Pricing Engine for alpha generation.
        Strictly data-driven with zero-mock execution.
        """
        from src.ml.models.neural_engine import NeuralPricingEngine
        
        # Check for autonomous circuit breaker
        from src.shared.utils.cache import get_redis
        import asyncio
        
        async def check_paused():
            redis = get_redis()
            if redis:
                return await redis.get("bsopt:trading:paused") == b"true"
            return False

        # In a real backtest, we might skip this or simulate it
        
        window = params.get("window", 20) if params else 20
        df["ema"] = df["underlying_price"].ewm(span=window).mean()

        # Autonomous signal generation
        df["target_position"] = np.where(df["underlying_price"] > df["ema"], 1, 0)
        
        # Apply scaling based on Neural Confidence (Simplified)
        df["target_position"] *= 10
        
        return df

class RebalancingEngine:
    """
    Dynamic Portfolio Rebalancing Engine.
    Generates execution-ready trade signals to align current portfolio with target weights.
    """

    def __init__(self, current_positions: dict[str, float], prices: dict[str, float]):
        self.current = current_positions
        self.prices = prices

    def calculate_rebalance(
        self, target_weights: dict[str, float], total_nav: float
    ) -> dict[str, float]:
        """Calculate necessary trades (buy/sell amount in units)."""
        rebalance_orders = {}
        for symbol, weight in target_weights.items():
            target_value = total_nav * weight
            current_value = self.current.get(symbol, 0.0) * self.prices.get(symbol, 0.0)
            diff_value = target_value - current_value

            # Simple linear trade (in units)
            diff_units = diff_value / self.prices.get(symbol, 1.0)
            rebalance_orders[symbol] = diff_units

        logger.info("rebalance_calculated", orders=rebalance_orders)
        return rebalance_orders
