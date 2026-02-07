"""
Machine Learning Performance Metrics
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate standard and weighted regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    # 🚀 OPTIMIZATION: Weighted MSE to penalize errors on high-premium options more
    weights = np.maximum(y_true, 1.0)
    wmse = np.average((y_true - y_pred) ** 2, weights=weights)

    return {
        "rmse": float(np.sqrt(mse)),
        "wrmse": float(np.sqrt(wmse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "max_pe": float(np.max(np.abs(y_true - y_pred) / np.maximum(y_true, 1e-5))),
        "r2": float(r2_score(y_true, y_pred)),
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

    drawdown = (equity_curve - running_max) / running_max
    return float(np.min(drawdown))
