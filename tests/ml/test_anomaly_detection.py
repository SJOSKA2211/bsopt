import pytest
import numpy as np
from src.aiops.isolation_forest_detector import IsolationForestDetector # Assuming this path

def test_isolation_forest_detector_init():
    """Test initialization of IsolationForestDetector."""
    detector = IsolationForestDetector(contamination=0.1)
    assert detector.contamination == 0.1
    assert detector.model is None

def test_isolation_forest_detector_fit_predict_univariate():
    """Test fitting and prediction for univariate data."""
    # Normal data with a few clear outliers
    data = np.array([
        1.0, 1.1, 1.05, 0.9, 1.2, 1.0, 1.15, 1.0, 1.0, 1.0, 1.0, 1.0,
        10.0,  # Outlier
        1.0, 0.95, 1.0, 1.1, 1.0, 1.0, 1.0,
        -5.0   # Outlier
    ]).reshape(-1, 1)

    detector = IsolationForestDetector(contamination=0.15)
    anomalies = detector.fit_predict(data)

    # Expected anomalies: -1 for outliers, 1 for inliers
    # The exact output can vary slightly due to IsolationForest's stochastic nature,
    # but the clear outliers should be detected.
    assert len(anomalies) == len(data)
    # Finding the indices of the original outliers in the modified data
    outlier_10_index = np.where(data == 10.0)[0][0]
    outlier_neg5_index = np.where(data == -5.0)[0][0]

    assert anomalies[outlier_10_index] == -1 # 10.0 is an outlier
    assert anomalies[outlier_neg5_index] == -1 # -5.0 is an outlier
    assert np.sum(anomalies == -1) >= 2 # At least the two clear outliers

def test_isolation_forest_detector_no_outliers():
    """Test behavior with data containing no obvious outliers."""
    data = np.array([1.0, 1.1, 1.05, 0.9, 1.2, 1.0, 1.15]).reshape(-1, 1)
    detector = IsolationForestDetector(contamination=0.1)
    anomalies = detector.fit_predict(data)
    assert len(anomalies) == len(data)
    # With contamination=0.1, it should still try to find 10% outliers,
    # but for such small, clean data, it might return all inliers or one outlier
    # depending on internal thresholds. We mostly care it doesn't crash.
    # A more robust check might be to assert that the sum of anomalies is mostly inliers.
    assert np.sum(anomalies == 1) > 0

def test_isolation_forest_detector_empty_data():
    """Test with empty input data."""
    data = np.array([]).reshape(0, 1)
    detector = IsolationForestDetector(contamination=0.1)
    with pytest.raises(ValueError, match="Expected 2D array, got 0D array instead"):
        detector.fit_predict(data)

def test_isolation_forest_detector_fit_predict_series_data():
    """Test with time series like data (just values, no timestamps)."""
    data = np.array([
        100, 101, 102, 100, 105, 103,
        200, # Spike
        100, 101, 102,
        50   # Dip
    ]).reshape(-1, 1)
    detector = IsolationForestDetector(contamination=0.15)
    anomalies = detector.fit_predict(data)
    assert len(anomalies) == len(data)
    assert anomalies[6] == -1 # 200 is an outlier
    assert anomalies[10] == -1 # 50 is an outlier
    assert np.sum(anomalies == -1) >= 2

def test_isolation_forest_detector_not_fitted_predict_raises_error():
    """Test that predict raises an error if model is not fitted."""
    detector = IsolationForestDetector(contamination=0.1)
    data = np.array([1.0, 2.0]).reshape(-1, 1)
    with pytest.raises(RuntimeError, match="Isolation Forest model has not been fitted yet."):
        detector.predict(data)

def test_isolation_forest_detector_invalid_contamination():
    """Test that Initialization with invalid contamination raises ValueError."""
    with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5."):
        IsolationForestDetector(contamination=0.0)
    with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5."):
        IsolationForestDetector(contamination=0.5)
    with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5."):
        IsolationForestDetector(contamination=-0.1)
    with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5."):
        IsolationForestDetector(contamination=1.0)

def test_isolation_forest_detector_fit_predict_1d_data():
    """Test fitting and prediction with 1D input data."""
    data = np.array([1.0, 1.1, 10.0, 0.9, -5.0]) # 1D array
    detector = IsolationForestDetector(contamination=0.2)
    anomalies = detector.fit_predict(data)
    assert len(anomalies) == len(data)
    assert np.sum(anomalies == -1) >= 1 # Expect at least one outlier

def test_isolation_forest_detector_predict_1d_data():
    """Test prediction with 1D input data after fitting with 2D data."""
    fit_data = np.array([1.0, 1.1, 1.05, 0.9, 1.2, 1.0, 1.15, 1.0, 1.0, 1.0, 1.0, 1.0,
                         10.0,
                         1.0, 0.95, 1.0, 1.1, 1.0, 1.0, 1.0,
                         -5.0]).reshape(-1, 1)
    detector = IsolationForestDetector(contamination=0.15)
    detector.fit_predict(fit_data) # Fit with 2D data

    predict_data = np.array([1.0, 1.1, 10.0]) # 1D array for prediction
    anomalies = detector.predict(predict_data)
    assert len(anomalies) == len(predict_data)
    assert anomalies[2] == -1 # 10.0 should be an outlier

