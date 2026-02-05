import pytest
import numpy as np
import torch
from src.aiops.transformer_detector import TransformerAnomalyDetector

def test_transformer_detector_init():
    detector = TransformerAnomalyDetector(input_dim=10, threshold=0.1)
    assert detector.input_dim == 10
    assert detector.threshold == 0.1
    assert not detector.is_fitted

def test_transformer_detector_detect_unfitted():
    detector = TransformerAnomalyDetector(input_dim=10)
    data = np.random.rand(5, 10)
    result = detector.detect(data)
    assert "is_anomaly" in result
    assert "score" in result
    assert "culprit_name" in result

def test_transformer_detector_train_and_detect():
    detector = TransformerAnomalyDetector(input_dim=10)
    train_data = np.random.rand(100, 10)
    detector.train_on_data(train_data, epochs=2)
    assert detector.is_fitted
    
    # Detect anomaly
    test_data = np.random.rand(5, 10)
    result = detector.detect(test_data)
    assert result["is_anomaly"] in [True, False]

def test_transformer_detector_3d_data():
    detector = TransformerAnomalyDetector(input_dim=10)
    train_data = np.random.rand(10, 20, 10) # (Batch, Seq, Feat)
    detector.train_on_data(train_data, epochs=1)
    
    test_data = np.random.rand(2, 20, 10)
    result = detector.detect(test_data)
    assert len(result["feature_errors"]) == 10
