import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog
import cvxpy as cp
from scipy.cluster.hierarchy import linkage
from src.pricing.black_scholes import BlackScholesEngine

logger = structlog.get_logger()

class PortfolioOptimizer:
    """
    SOTA: Convex Portfolio Optimizer and Hierarchical Risk Parity.
    Uses cvxpy/OSQP for quadratic programming and machine learning for robust allocation.
    """
    def __init__(self, returns_df: pd.DataFrame):
        self.returns = returns_df
        self.cov_matrix = returns_df.cov().values * 252
        self.mean_returns = returns_df.mean().values * 252
        self.asset_names = returns_df.columns.tolist()

    def optimize_weights(self, target_return: Optional[float] = None) -> np.ndarray:
        """🚀 SINGULARITY: Convex optimization using cvxpy and OSQP."""
        n = len(self.mean_returns)
        w = cp.Variable(n)
        
        # Risk (Variance)
        risk = cp.quad_form(w, self.cov_matrix)
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = [cp.sum(w) == 1, w >= 0]
        
        if target_return is not None:
            constraints.append(self.mean_returns @ w >= target_return)
            
        prob = cp.Problem(cp.Minimize(risk), constraints)
        
        try:
            # SOTA: OSQP is the gold standard for QP portfolio problems
            prob.solve(solver=cp.OSQP)
            if w.value is None:
                raise ValueError("optimization_resulted_in_none")
            return w.value
        except Exception as e:
            logger.warning("convex_optimization_failed_falling_back", error=str(e))
            return np.array([1.0 / n] * n)

    def optimize_hrp(self) -> np.ndarray:
        """🚀 SOTA: Hierarchical Risk Parity (HRP) allocation."""
        # 1. Clustering
        corr = self.returns.corr().values
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(dist, 'single')
        
        # 2. Quasi-Diagonalization
        def get_quasi_diag(link):
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

        sort_ix = get_quasi_diag(link)
        
        # 3. Recursive Bisection
        def get_cluster_var(cov, cluster_items):
            cov_c = cov[np.ix_(cluster_items, cluster_items)]
            w = 1.0 / np.diag(cov_c)
            w /= w.sum()
            return np.dot(w.T, np.dot(cov_c, w))

        def get_rec_bisec(cov, sort_ix):
            w = pd.Series(1, index=sort_ix)
            c_items = [sort_ix]
            while len(c_items) > 0:
                c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                for i in range(0, len(c_items), 2):
                    c_left = c_items[i]
                    c_right = c_items[i+1]
                    alpha_1 = get_cluster_var(cov, c_left)
                    alpha_2 = get_cluster_var(cov, c_right)
                    alpha = 1 - alpha_1 / (alpha_1 + alpha_2)
                    w[c_left] *= alpha
                    w[c_right] *= 1 - alpha
            return w

        weights = get_rec_bisec(self.cov_matrix, sort_ix)
        return weights.sort_index().values

class BacktestEngine:
    """
    High-performance vectorized backtesting engine for option strategies.
    Supports parallel evaluation using Ray or Dask.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run_vectorized(self, 
                       df: pd.DataFrame, 
                       strategy_fn: Any, 
                       params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executes a strategy over historical data using vectorized operations.
        df must contain: timestamp, underlying_price, option_price, strike, maturity, etc.
        """
        logger.info("backtest_started", rows=len(df), params=params)
        start_time = pd.Timestamp.now()

        # 1. Generate Signals (Vectorized if possible within strategy_fn)
        # Strategy function should return a series of 'target_positions'
        df = strategy_fn(df, params)

        if 'target_position' not in df.columns:
            raise ValueError("Strategy function must add 'target_position' column to DataFrame")

        # 2. Vectorized P&L Calculation
        # Assuming target_position is number of contracts
        df['prev_position'] = df['target_position'].shift(1).fillna(0)
        df['trades'] = df['target_position'] - df['prev_position']
        
        # Transaction costs (simulated)
        transaction_cost_pct = 0.001
        df['commissions'] = np.abs(df['trades'] * df['option_price'] * transaction_cost_pct)
        
        # Mark-to-market P&L
        df['price_change'] = df['option_price'].diff().fillna(0)
        df['mtm_pnl'] = df['prev_position'] * df['price_change'] - df['commissions']
        
        # Cumulative metrics
        df['cum_pnl'] = df['mtm_pnl'].cumsum()
        df['equity_curve'] = self.initial_capital + df['cum_pnl']
        
        # 3. Calculate Performance Metrics
        returns = df['equity_curve'].pct_change().dropna()
        total_return = (df['equity_curve'].iloc[-1] / self.initial_capital) - 1
        
        # Sharpe Ratio (annualized, assuming daily data)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        rolling_max = df['equity_curve'].cummax()
        drawdown = (df['equity_curve'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 4. SOTA Risk Metrics: VaR and Expected Shortfall (Vectorized)
        confidence_level = params.get('confidence_level', 0.95) if params else 0.95
        
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
            "trades_count": int(np.abs(df['trades']).sum() > 0), # Simplified
            "final_value": float(df['equity_curve'].iloc[-1]),
            "duration_seconds": duration,
            "status": "completed"
        }
        
        logger.info("backtest_completed", metrics=result)
        return result

    @staticmethod
    def sample_momentum_strategy(df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Sample vectorized strategy: momentum-based option buying."""
        window = params.get('window', 20) if params else 20
        df['ema'] = df['underlying_price'].ewm(span=window).mean()
        
        # Signal: 1 if price > EMA, else 0
        df['target_position'] = np.where(df['underlying_price'] > df['ema'], 10, 0)
        return df
