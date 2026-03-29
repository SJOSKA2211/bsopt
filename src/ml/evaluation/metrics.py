"""
Machine Learning Performance Metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive regression performance metrics.
    Includes weighted RMSE to account for relative impact on high-premium options.
    """
    mse = mean_squared_error(y_true, y_pred)

    # Weighted MSE: Use higher premiums as higher importance
    weights = np.maximum(y_true, 1.0)
    wmse = np.average((y_true - y_pred) ** 2, weights=weights)

    # Handle R2 edge case (constant target)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    return {
        "rmse": float(np.sqrt(mse)),
        "wrmse": float(np.sqrt(wmse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "max_pe": float(np.max(np.abs(y_true - y_pred) / np.maximum(y_true, 1e-5))),
        "r2": r2,
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate standard classification metrics for fallback scenarios."""
    # Threshold if coming from probabilities or soft signals
    y_true_bin = (y_true > 0.5).astype(int)
    y_pred_bin = (y_pred > 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }

def calculate_pricing_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate model pricing bias (mean error)."""
    return float(np.mean(y_pred - y_true))

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    Assumes daily returns if not specified.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns) - risk_free_rate / 252
    std_return = np.std(returns)

    if std_return < 1e-9:
        return 0.0

    return float(mean_return / std_return * np.sqrt(252))

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate the maximum peak-to-trough drawdown of an equity curve.
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    # Avoid division by zero
    running_max = np.maximum(running_max, 1e-9)
    drawdown = (running_max - equity_curve) / running_max
    return float(np.max(drawdown))

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sortino Ratio.
    Uses only downside deviation (negative returns) for risk measurement.
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / 252
    mean_excess = np.mean(excess_returns)

    # Downside deviation: std of negative returns only
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) < 1:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = np.std(negative_returns)
    if downside_std < 1e-9:
        return float("inf") if mean_excess > 0 else 0.0

    return float(mean_excess / downside_std * np.sqrt(252))

class ModelScorecard:
    """
    Unified performance scorecard combining regression and financial metrics.
     OPTIMIZED: Holistic model evaluation.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, returns: np.ndarray | None = None):
        self.regression_metrics = calculate_regression_metrics(y_true, y_pred)
        self.pricing_bias = calculate_pricing_bias(y_true, y_pred)

        if returns is not None:
            self.sharpe_ratio = calculate_sharpe_ratio(returns)
            self.sortino_ratio = calculate_sortino_ratio(returns)
            self.max_drawdown = calculate_max_drawdown(
                np.cumsum(returns) + 1.0
            )  # Cumulative equity
        else:
            self.sharpe_ratio = 0.0
            self.sortino_ratio = 0.0
            self.max_drawdown = 0.0

    def to_dict(self) -> dict:
        return {
            **self.regression_metrics,
            "pricing_bias": self.pricing_bias,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "score": self.calculate_composite_score(),
        }

    def calculate_composite_score(self) -> float:
        """Calculates a single score representing overall model quality (0 to 1)."""
        r2 = max(0, self.regression_metrics["r2"])
        # Penalize bias and drawdown, reward Sharpe
        sharpe_norm = min(max(self.sharpe_ratio / 3.0, 0), 1.0)  # Assume 3.0 is excellent
        mdd_penalty = min(abs(self.max_drawdown), 1.0)

        return float(0.4 * r2 + 0.4 * sharpe_norm + 0.2 * (1.0 - mdd_penalty))
