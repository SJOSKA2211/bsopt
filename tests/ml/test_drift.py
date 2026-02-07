from unittest.mock import patch

import numpy as np
import pytest

from src.ml.drift import calculate_ks_test, calculate_psi


@patch("src.ml.drift.logger")
@patch("src.ml.drift.KS_TEST_SCORE")
def test_ks_test_instrumentation(mock_gauge, mock_logger):
    expected = np.random.normal(0, 1, 100)
    actual = np.random.normal(0, 1, 100)

    calculate_ks_test(expected, actual)

    # Verify logger was called
    assert mock_logger.info.called
    # Verify prometheus gauge was updated
    mock_gauge.set.assert_called_once()


def test_calculate_ks_test_no_drift():
    """Verify that KS test returns high p-value when distributions are identical."""
    np.random.seed(42)
    expected = np.random.normal(0, 1, 2000)
    actual = np.random.normal(0, 1, 2000)

    statistic, p_value = calculate_ks_test(expected, actual)
    # p-value should be high (usually > 0.05 means no significant difference)
    assert p_value > 0.05


def test_calculate_ks_test_significant_drift():
    """Verify that KS test returns low p-value when distributions are different."""
    np.random.seed(42)
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(5, 1, 1000)

    statistic, p_value = calculate_ks_test(expected, actual)
    # p-value should be very low
    assert p_value < 0.01


@patch("src.ml.drift.logger")
@patch("src.ml.drift.DATA_DRIFT_SCORE")
def test_drift_instrumentation(mock_gauge, mock_logger):
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    actual = np.array([0.1, 0.2, 0.3, 0.4])

    calculate_psi(expected, actual, buckets=4)

    # Verify logger was called
    assert mock_logger.info.called
    # Verify prometheus gauge was updated
    mock_gauge.set.assert_called_once()


def test_calculate_psi_no_drift():
    """Verify that PSI is 0 when distributions are identical."""
    expected = np.array([0.1, 0.2, 0.3, 0.4])
    actual = np.array([0.1, 0.2, 0.3, 0.4])

    psi_score = calculate_psi(expected, actual, buckets=4)
    assert psi_score == pytest.approx(0.0)


def test_calculate_psi_significant_drift():
    """Verify that PSI detects significant drift."""
    # Reference data: normal distribution centered at 0
    expected = np.random.normal(0, 1, 1000)
    # Actual data: normal distribution centered at 1 (drifted)
    actual = np.random.normal(1, 1, 1000)

    psi_score = calculate_psi(expected, actual, buckets=10)
    # PSI > 0.2 is usually considered significant drift
    assert psi_score > 0.2


def test_calculate_psi_zero_bin_handling():
    """Verify that PSI calculation handles empty buckets gracefully."""
    expected = np.array([1, 2, 3, 4, 5])
    actual = np.array([1, 2, 0, 4, 5])  # One bucket is empty in 'actual'

    psi_score = calculate_psi(expected, actual, buckets=5)
    # It should not raise ZeroDivisionError or return NaN
    assert not np.isnan(psi_score)
    assert psi_score > 0


def test_performance_drift_monitor():
    """Verify that performance drift is correctly detected."""
    from src.ml.drift import PerformanceDriftMonitor

    # Initialize monitor with a window size of 3
    monitor = PerformanceDriftMonitor(window_size=3, threshold=0.1)

    # Add initial metrics (baseline)
    monitor.add_metric(0.85)  # Run 1
    monitor.add_metric(0.86)  # Run 2
    monitor.add_metric(0.84)  # Run 3

    # No drift yet
    assert monitor.detect_drift(0.84) is False

    # Significant degradation (baseline average is 0.85, 0.70 is > 0.1 lower)
    assert monitor.detect_drift(0.70) is True

    # Improvement should not trigger drift
    assert monitor.detect_drift(0.95) is False


@patch("src.ml.drift.PERFORMANCE_DRIFT_ALERT")
def test_performance_drift_alert_instrumentation(mock_alert):
    """Verify that PerformanceDriftMonitor updates the Prometheus alert gauge."""
    from src.ml.drift import PerformanceDriftMonitor

    monitor = PerformanceDriftMonitor(window_size=2, threshold=0.1)
    monitor.add_metric(0.9)
    monitor.add_metric(0.9)

    # Trigger drift
    monitor.detect_drift(0.5)
    mock_alert.set.assert_called_with(1)

    # No drift
    monitor.detect_drift(0.9)
    mock_alert.set.assert_called_with(0)


def test_performance_drift_insufficient_history():
    """Verify that drift detection is skipped when history is insufficient."""
    from src.ml.drift import PerformanceDriftMonitor

    monitor = PerformanceDriftMonitor(window_size=5)
    monitor.add_metric(0.8)

    # History size is 1, window size is 5 -> should return False
    assert monitor.detect_drift(0.7) is False
