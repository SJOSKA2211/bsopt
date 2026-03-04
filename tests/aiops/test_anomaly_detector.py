import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.aiops.anomaly_detector import AnomalyDetector

def test_anomaly_detector_init():
    detector = AnomalyDetector(contamination=0.1)
    assert not detector.is_fitted
    assert detector.model.contamination == 0.1

def test_anomaly_detector_train_pandas():
    detector = AnomalyDetector()
    data = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    
    with patch.object(detector.model, "fit") as mock_fit:
        detector.train(data)
        assert detector.is_fitted
        assert len(detector.columns) == 2
        mock_fit.assert_called_once()

def test_anomaly_detector_train_numpy():
    detector = AnomalyDetector()
    data = np.random.rand(20, 2)
    
    with patch.object(detector.model, "fit") as mock_fit:
        detector.train(data)
        assert detector.is_fitted
        assert len(detector.columns) == 2
        mock_fit.assert_called_once()

def test_anomaly_detector_detect_pandas():
    detector = AnomalyDetector()
    detector.is_fitted = True
    detector.columns = ["a"]
    
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    # Mock model
    detector.model.predict = MagicMock(return_value=np.array([-1, 1, -1]))
    detector.model.decision_function = MagicMock(return_value=np.array([-0.5, 0.5, -0.6]))
    
    anomalies = detector.detect(data)
    assert len(anomalies) == 2
    assert anomalies[0]["index"] == 0
    assert anomalies[0]["metrics"]["a"] == 1.0
    assert anomalies[1]["index"] == 2

def test_anomaly_detector_detect_numpy():
    detector = AnomalyDetector()
    detector.is_fitted = True
    detector.columns = ["feat_0"]
    
    data = np.array([[1.0], [2.0]])
    detector.model.predict = MagicMock(return_value=np.array([-1, 1]))
    detector.model.decision_function = MagicMock(return_value=np.array([-0.5, 0.5]))
    
    anomalies = detector.detect(data)
    assert len(anomalies) == 1
    assert anomalies[0]["index"] == 0
    assert anomalies[0]["metrics"]["feat_0"] == 1.0

def test_unfitted_error():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.detect(np.array([[1]]))
