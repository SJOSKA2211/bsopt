import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.ml.drift import calculate_psi

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
    actual = np.array([1, 2, 0, 4, 5]) # One bucket is empty in 'actual'
    
    psi_score = calculate_psi(expected, actual, buckets=5)
    # It should not raise ZeroDivisionError or return NaN
    assert not np.isnan(psi_score)
    assert psi_score > 0
