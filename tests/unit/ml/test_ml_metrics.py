import numpy as np

from src.ml.evaluation.metrics import (
    ModelScorecard,
    calculate_max_drawdown,
    calculate_pricing_bias,
    calculate_regression_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
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


def test_sharpe_ratio_positive_returns():
    """Sharpe ratio should be positive for consistently positive returns."""
    returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 0


def test_sharpe_ratio_edge_cases():
    """Test edge cases: single element and zero std."""
    # Single element should return 0
    assert calculate_sharpe_ratio(np.array([0.01])) == 0.0
    # Zero volatility case
    assert calculate_sharpe_ratio(np.array([0.0, 0.0, 0.0])) == 0.0


def test_sortino_ratio_positive_returns():
    """Sortino should be higher than Sharpe when only upside volatility exists."""
    returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
    sortino = calculate_sortino_ratio(returns)
    sharpe = calculate_sharpe_ratio(returns)
    # With only positive returns, sortino may return inf or be higher than sharpe
    assert sortino >= sharpe or sortino == float("inf")


def test_sortino_ratio_with_downside():
    """Sortino should be finite when there are negative returns."""
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    sortino = calculate_sortino_ratio(returns)
    assert sortino != float("inf")
    assert isinstance(sortino, float)


def test_sortino_ratio_edge_cases():
    """Test edge cases for Sortino ratio."""
    # Single element should return 0
    assert calculate_sortino_ratio(np.array([0.01])) == 0.0


def test_max_drawdown_calculation():
    """Max drawdown should correctly identify the largest peak-to-trough decline."""
    # Equity curve: starts at 100, peaks at 120, drops to 90, recovers to 110
    equity = np.array([100, 110, 120, 100, 90, 100, 110])
    mdd = calculate_max_drawdown(equity)
    # Max drawdown from 120 to 90 = 30/120 = 0.25
    assert abs(mdd - 0.25) < 0.01


def test_max_drawdown_no_drawdown():
    """Max drawdown should be 0 for monotonically increasing equity."""
    equity = np.array([100, 110, 120, 130, 140])
    mdd = calculate_max_drawdown(equity)
    assert mdd == 0.0


def test_max_drawdown_edge_cases():
    """Test edge cases."""
    # Single element should return 0
    assert calculate_max_drawdown(np.array([100])) == 0.0


def test_model_scorecard_no_returns():
    """ModelScorecard should work without returns data."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])

    scorecard = ModelScorecard(y_true, y_pred)
    result = scorecard.to_dict()

    assert "rmse" in result
    assert "mae" in result
    assert "sharpe_ratio" in result
    assert "sortino_ratio" in result
    assert "max_drawdown" in result
    assert result["sharpe_ratio"] == 0.0
    assert result["sortino_ratio"] == 0.0


def test_model_scorecard_with_returns():
    """ModelScorecard should calculate financial metrics when returns provided."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
    returns = np.array([0.01, -0.02, 0.03, 0.01, 0.02])

    scorecard = ModelScorecard(y_true, y_pred, returns=returns)
    result = scorecard.to_dict()

    assert result["sharpe_ratio"] != 0.0
    assert result["sortino_ratio"] != 0.0
    assert "score" in result
    assert 0 <= result["score"] <= 1