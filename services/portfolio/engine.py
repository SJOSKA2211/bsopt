from typing import Any

import numpy as np
import pandas as pd
import ray
import structlog
from scipy.cluster.hierarchy import linkage

from core.shared.distributed import RayOrchestrator

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
    # ... (init and optimize_weights stay same)

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
        from core.trading.backtesting.kernel import calculate_metrics_kernel, run_simulation_kernel

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

        # 3. Calculate Performance Metrics using optimized kernel
        total_return, sharpe, max_drawdown = calculate_metrics_kernel(
            equity_curve, self.initial_capital
        )

        # 4. OPTIMIZED Risk Metrics: VaR and Expected Shortfall (Vectorized)
        confidence_level = params.get("confidence_level", 0.95) if params else 0.95

        # Calculate periodic returns from equity curve
        returns = pd.Series(equity_curve).pct_change().dropna()

        # Historical VaR
        # Sort returns and find the percentile
        var_95 = np.percentile(returns, (1 - confidence_level) * 100) if not returns.empty else 0.0

        # Expected Shortfall (Average of returns worse than VaR)
        es_95 = returns[returns <= var_95].mean() if not returns.empty else 0.0

        duration = (pd.Timestamp.now() - start_time).total_seconds()

        result = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "es_95": float(es_95),
            "trades_count": int(np.abs(df["trades"]).sum() > 0),  # Simplified
            "final_value": float(df["equity_curve"].iloc[-1]),
            "duration_seconds": duration,
            "status": "completed",
        }

        logger.info("backtest_completed", metrics=result)
        return result

    @staticmethod
    def sample_momentum_strategy(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """Sample vectorized strategy: momentum-based option buying."""
        window = params.get("window", 20) if params else 20
        df["ema"] = df["underlying_price"].ewm(span=window).mean()

        # Signal: 1 if price > EMA, else 0
        df["target_position"] = np.where(df["underlying_price"] > df["ema"], 10, 0)
        return df
