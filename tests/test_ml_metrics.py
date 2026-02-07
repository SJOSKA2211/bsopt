import numpy as np

from src.ml.evaluation.metrics import (
    calculate_pricing_bias,
    calculate_regression_metrics,
)
from tests.test_utils import assert_equal


def test_regression_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert_equal(metrics["mae"], 1.0)


def test_pricing_bias():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 21.0, 31.0])

    bias = calculate_pricing_bias(y_true, y_pred)
    assert_equal(bias, 1.0)
