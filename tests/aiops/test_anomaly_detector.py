import numpy as np
import pandas as pd
import pytest
import torch

from src.ml.aiops.anomaly_detector import AnomalyDetector

#  Initialization Tests 


def test_anomaly_detector_init_defaults():
    detector = AnomalyDetector()
    assert detector.engine == "isolation_forest"
    assert not detector.is_fitted


def test_anomaly_detector_init_engines():
    # Isolation Forest
    if_detector = AnomalyDetector(engine="isolation_forest", contamination=0.1)
    assert if_detector.model.contamination == 0.1

    # Autoencoder
    ae_detector = AnomalyDetector(engine="autoencoder", input_dim=5, latent_dim=2)
    assert ae_detector.input_dim == 5
    assert ae_detector.latent_dim == 2

    # Transformer
    tf_detector = AnomalyDetector(engine="transformer", input_dim=5, threshold=0.1)
    assert tf_detector.input_dim == 5
    assert tf_detector.threshold == 0.1


def test_anomaly_detector_invalid_engine():
    with pytest.raises(ValueError, match="Unknown anomaly detection engine"):
        AnomalyDetector(engine="invalid_engine")


#  Training & Detection Tests 


def test_isolation_forest_workflow():
    detector = AnomalyDetector(engine="isolation_forest")
    data = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})

    detector.train(data)
    assert detector.is_fitted
    assert len(detector.columns) == 2

    anomalies = detector.detect(data)
    assert isinstance(anomalies, list)


def test_autoencoder_workflow():
    input_dim = 4
    detector = AnomalyDetector(engine="autoencoder", input_dim=input_dim, latent_dim=2)
    data = np.random.rand(50, input_dim)

    detector.train(data, epochs=2)
    assert detector.is_fitted
    assert detector.threshold is not None

    anomalies = detector.detect(data)
    assert isinstance(anomalies, list)
    if anomalies:
        assert anomalies[0]["type"] == "reconstruction_error"


def test_transformer_workflow():
    input_dim = 4
    detector = AnomalyDetector(engine="transformer", input_dim=input_dim)
    # (Batch, Seq, Feat)
    data = np.random.rand(5, 10, input_dim)

    detector.train(data, epochs=2)
    assert detector.is_fitted

    results = detector.detect(data)
    assert isinstance(results, list)


#  Error Handling & Edge Cases 


def test_unfitted_error():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.detect(np.random.rand(5, 1))


def test_empty_data_handling():
    detector = AnomalyDetector()
    empty_df = pd.DataFrame()
    detector.train(empty_df)
    assert not detector.is_fitted

    detector.is_fitted = True
    assert detector.detect(empty_df) == []


def test_contamination_validation():
    with pytest.raises(ValueError, match="Contamination must be between 0 and 0.5"):
        AnomalyDetector(contamination=0.6)