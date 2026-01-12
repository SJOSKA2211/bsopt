import pytest
import numpy as np
import pandas as pd
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector

def test_anomaly_detector_training():
    """Verify that the detector trains on historical data."""
    detector = TimeSeriesAnomalyDetector()
    
    # Create normal historical data (sine wave + small noise)
    t = np.linspace(0, 10, 100)
    data = pd.DataFrame({
        "latency": np.sin(t) + 1.0 + np.random.normal(0, 0.1, 100),
        "cpu_usage": np.cos(t) + 1.0 + np.random.normal(0, 0.1, 100)
    })
    
    detector.train(data)
    assert detector.is_fitted is True

def test_anomaly_detector_detection():
    """Verify that the detector identifies outliers."""
    detector = TimeSeriesAnomalyDetector(contamination=0.1)
    
    # Train on normal data
    normal_data = pd.DataFrame({
        "metric": np.random.normal(10, 1, 100)
    })
    detector.train(normal_data)
    
    # Detect on data with a clear anomaly
    test_data = pd.DataFrame({
        "metric": [10.1, 9.9, 10.2, 50.0, 9.8] # 50.0 is an anomaly
    })
    
    anomalies = detector.detect(test_data)
    
    assert len(anomalies) >= 1
    # Check if the extreme value (index 3) was caught
    anomaly_indices = [a["index"] for a in anomalies]
    assert 3 in anomaly_indices
    assert anomalies[0]["metrics"]["metric"] == 50.0

def test_anomaly_detector_unfitted_error():
    """Verify that detect() raises an error if model is not fitted."""
    detector = TimeSeriesAnomalyDetector()
    with pytest.raises(RuntimeError, match="Model must be trained"):
        detector.detect(pd.DataFrame({"a": [1]}))
