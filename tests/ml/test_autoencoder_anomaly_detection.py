import pytest
import numpy as np
from src.aiops.autoencoder_detector import AutoencoderDetector, Autoencoder # Import actual Autoencoder

def test_autoencoder_detector_init():
    """Test initialization of AutoencoderDetector."""
    input_dim = 10
    latent_dim = 2
    epochs = 5
    threshold_multiplier = 2.0
    detector = AutoencoderDetector(input_dim, latent_dim, epochs, threshold_multiplier)
    assert detector.input_dim == input_dim
    assert detector.latent_dim == latent_dim
    assert detector.epochs == epochs
    assert detector.threshold_multiplier == threshold_multiplier
    assert isinstance(detector.model, Autoencoder) # Assert against actual Autoencoder
    assert detector.threshold is None

def test_autoencoder_detector_fit_predict_multivariate():
    """Test fitting and prediction for multivariate data."""
    input_dim = 5
    latent_dim = 2
    epochs = 1
    threshold_multiplier = 2.0
    
    # Generate some normal data
    np.random.seed(42)
    normal_data = np.random.rand(100, input_dim) * 0.1
    
    # Introduce some anomalies
    anomaly_data = np.random.rand(10, input_dim) * 10.0
    
    data = np.vstack((normal_data, anomaly_data))
    
    detector = AutoencoderDetector(input_dim, latent_dim, epochs, threshold_multiplier, verbose=False)
    anomalies = detector.fit_predict(data)
    
    assert len(anomalies) == len(data)
    
    # Check that a significant portion of the introduced anomalies are detected
    # and a significant portion of normal data is not detected as anomaly
    num_normal = len(normal_data)
    num_anomaly = len(anomaly_data)
    
    detected_anomalies_count = np.sum(anomalies[num_normal:] == -1)
    false_positives_count = np.sum(anomalies[:num_normal] == -1)
    
    # This might need adjustment based on MockAutoencoder's performance
    assert detected_anomalies_count > (num_anomaly * 0.7) # Detect most true anomalies
    assert false_positives_count < (num_normal * 0.1) # Few false positives

def test_autoencoder_detector_empty_data():
    """Test with empty input data."""
    input_dim = 5
    latent_dim = 2
    epochs = 1
    threshold_multiplier = 2.0
    data = np.array([]).reshape(0, input_dim)
    detector = AutoencoderDetector(input_dim, latent_dim, epochs, threshold_multiplier)
    with pytest.raises(ValueError, match="Input data for Autoencoder must not be empty."):
        detector.fit_predict(data)

def test_autoencoder_detector_not_fitted_predict_raises_error():
    """Test that predict raises an error if model is not fitted."""
    input_dim = 5
    latent_dim = 2
    epochs = 1
    threshold_multiplier = 2.0
    detector = AutoencoderDetector(input_dim, latent_dim, epochs, threshold_multiplier)
    data = np.random.rand(10, input_dim)
    with pytest.raises(RuntimeError, match="Autoencoder model has not been fitted yet."):
        detector.predict(data)

def test_autoencoder_detector_predict_multivariate():
    """Test prediction for multivariate data after fitting."""
    input_dim = 5
    latent_dim = 2
    epochs = 1
    threshold_multiplier = 2.0
    
    # Generate some normal data for fitting
    np.random.seed(42)
    normal_data = np.random.rand(100, input_dim) * 0.1
    
    detector = AutoencoderDetector(input_dim, latent_dim, epochs, threshold_multiplier, verbose=False)
    detector.fit(normal_data) # Fit the model
    
    # Generate data for prediction (some normal, some anomaly)
    predict_normal_data = np.random.rand(10, input_dim) * 0.1
    predict_anomaly_data = np.random.rand(5, input_dim) * 10.0
    
    predict_data = np.vstack((predict_normal_data, predict_anomaly_data))
    
    anomalies = detector.predict(predict_data)
    
    assert len(anomalies) == len(predict_data)
    
    num_predict_normal = len(predict_normal_data)
    num_predict_anomaly = len(predict_anomaly_data)
    
    detected_anomalies_count = np.sum(anomalies[num_predict_normal:] == -1)
    false_positives_count = np.sum(anomalies[:num_predict_normal] == -1)
    
    assert detected_anomalies_count > (num_predict_anomaly * 0.7) # Detect most true anomalies
    assert false_positives_count < (num_predict_normal * 0.1) # Few false positives
