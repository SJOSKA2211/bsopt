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
    """Calculate standard regression metrics for model evaluation."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def calculate_pricing_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate model pricing bias (mean error)."""
    return float(np.mean(y_pred - y_true))
