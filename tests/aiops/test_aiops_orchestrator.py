import pytest
from unittest.mock import patch

import numpy as np

# Assuming all these components are in src.aiops
from src.aiops.aiops_orchestrator import AIOpsOrchestrator # Assuming this path


# Mocking all dependencies for the orchestrator
@pytest.fixture
def mock_dependencies():
    with patch('src.aiops.aiops_orchestrator.PrometheusClient') as MockPrometheusClient, \
         patch('src.aiops.aiops_orchestrator.IsolationForestDetector') as MockIsolationForestDetector, \
         patch('src.aiops.aiops_orchestrator.AutoencoderDetector') as MockAutoencoderDetector, \
         patch('src.aiops.aiops_orchestrator.DataDriftDetector') as MockDataDriftDetector, \
         patch('src.aiops.aiops_orchestrator.DockerRemediator') as MockDockerRemediator, \
         patch('src.aiops.aiops_orchestrator.MLPipelineTrigger') as MockMLPipelineTrigger, \
         patch('src.aiops.aiops_orchestrator.RedisRemediator') as MockRedisRemediator, \
         patch('src.aiops.aiops_orchestrator.setup_logging') as MockSetupLogging, \
         patch('src.aiops.aiops_orchestrator.logger') as MockOrchestratorLogger:
        
        # Configure PrometheusClient mocks
        mock_prometheus_client_instance = MockPrometheusClient.return_value
        mock_prometheus_client_instance.get_5xx_error_rate.return_value = 0.01 # Default to low error
        mock_prometheus_client_instance.get_p95_latency.return_value = 0.1 # Default to low latency
        mock_prometheus_client_instance.get_historical_metric_data.return_value = np.array([1.0, 2.0, 3.0]).reshape(-1, 1) # Dummy data
        mock_prometheus_client_instance.get_historical_metric_data_multi.return_value = np.array([[1.0, 0.5], [2.0, 0.6], [3.0, 0.7]]) # Dummy data
        
        # Configure IsolationForestDetector mocks
        mock_isolation_forest_detector_instance = MockIsolationForestDetector.return_value
        mock_isolation_forest_detector_instance.fit_predict.return_value = [1] * 10 # Default to no anomalies
        
        # Configure AutoencoderDetector mocks
        mock_autoencoder_detector_instance = MockAutoencoderDetector.return_value
        mock_autoencoder_detector_instance.fit_predict.return_value = [1] * 10 # Default to no anomalies
        
        # Configure DataDriftDetector mocks
        mock_data_drift_detector_instance = MockDataDriftDetector.return_value
        mock_data_drift_detector_instance.detect_drift.return_value = (False, {}) # Default to no drift
        
        # Configure Remediators mocks
        mock_docker_remediator_instance = MockDockerRemediator.return_value
        mock_docker_remediator_instance.restart_service.return_value = True
        
        mock_ml_pipeline_trigger_instance = MockMLPipelineTrigger.return_value
        mock_ml_pipeline_trigger_instance.trigger_retraining.return_value = True
        
        mock_redis_remediator_instance = MockRedisRemediator.return_value
        mock_redis_remediator_instance.purge_cache.return_value = True
        
        yield {
            "PrometheusClient": MockPrometheusClient,
            "IsolationForestDetector": MockIsolationForestDetector,
            "AutoencoderDetector": MockAutoencoderDetector,
            "DataDriftDetector": MockDataDriftDetector,
            "DockerRemediator": MockDockerRemediator,
            "MLPipelineTrigger": MockMLPipelineTrigger,
            "RedisRemediator": MockRedisRemediator,
            "setup_logging": MockSetupLogging,
            "OrchestratorLogger": MockOrchestratorLogger,
        }

def test_aiops_orchestrator_init(mock_dependencies):
    """Test initialization of AIOpsOrchestrator."""
    config = {
        "check_interval_seconds": 1,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*",
        "autoencoder_input_dim": 10,  # Add these for AutoencoderDetector init
        "autoencoder_latent_dim": 2,
        "autoencoder_epochs": 5,
        "autoencoder_threshold_multiplier": 2.0,
    }
    orchestrator = AIOpsOrchestrator(config)
    assert orchestrator.config == config
    mock_dependencies["PrometheusClient"].assert_called_once_with(url="http://localhost:9090")
    mock_dependencies["IsolationForestDetector"].assert_called_once()
    mock_dependencies["AutoencoderDetector"].assert_called_once()
    mock_dependencies["DataDriftDetector"].assert_called_once()
    mock_dependencies["DockerRemediator"].assert_called_once()
    mock_dependencies["MLPipelineTrigger"].assert_called_once_with(config=config["ml_pipeline_config"]) # Fixed assertion
    mock_dependencies["RedisRemediator"].assert_called_once()
    mock_dependencies["setup_logging"].assert_called_once()


def test_aiops_orchestrator_run_no_anomalies(mock_dependencies):
    """Test the orchestrator loop with no anomalies detected."""
    config = {
        "check_interval_seconds": 0.01, # Short interval for testing
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*",
        "autoencoder_input_dim": 10,
        "autoencoder_latent_dim": 2,
    }
    orchestrator = AIOpsOrchestrator(config)

    # Run loop for a short period
    with patch('time.sleep') as mock_sleep:
        orchestrator.run(iterations=2)
        mock_sleep.assert_called_with(config["check_interval_seconds"])
    
    # Verify detection methods were called
    mock_dependencies["PrometheusClient"].return_value.get_5xx_error_rate.assert_called_with(service="api")
    mock_dependencies["PrometheusClient"].return_value.get_p95_latency.assert_called_with(service="api")
    # For IsolationForestDetector and AutoencoderDetector, assume they are called internally
    # with some data. For this test, we care that no remediation is triggered.
    
    # Verify no remediation actions were triggered
    mock_dependencies["DockerRemediator"].return_value.restart_service.assert_not_called()
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()
    mock_dependencies["RedisRemediator"].return_value.purge_cache.assert_not_called()


def test_aiops_orchestrator_run_high_5xx_error_remediation(mock_dependencies):
    """Test orchestrator triggers Docker restart on high 5xx errors."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "error_rate_threshold": 0.05, # Set threshold for this test
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    mock_dependencies["PrometheusClient"].return_value.get_5xx_error_rate.return_value = 0.1 # High error
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["DockerRemediator"].return_value.restart_service.assert_called_once_with("api") # Changed assertion
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()
    mock_dependencies["RedisRemediator"].return_value.purge_cache.assert_not_called()

def test_aiops_orchestrator_run_high_latency_remediation(mock_dependencies):
    """Test orchestrator triggers Docker restart on high latency."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "latency_threshold": 0.5, # Set threshold for this test
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    mock_dependencies["PrometheusClient"].return_value.get_p95_latency.return_value = 0.6 # High latency
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["DockerRemediator"].return_value.restart_service.assert_called_once_with("api") # Changed assertion
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()
    mock_dependencies["RedisRemediator"].return_value.purge_cache.assert_not_called()

def test_aiops_orchestrator_run_data_drift_remediation(mock_dependencies):
    """Test orchestrator triggers ML retraining on data drift."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "data_drift_detection_enabled": True, # Enable drift detection for this test
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    # Simulate data drift detected
    mock_dependencies["DataDriftDetector"].return_value.detect_drift.return_value = (True, {"drift_types": ["PSI_Drift"]}) 
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["DataDriftDetector"].return_value.detect_drift.assert_called_once()
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_called_once()
    mock_dependencies["DockerRemediator"].return_value.restart_service.assert_not_called()
    mock_dependencies["RedisRemediator"].return_value.purge_cache.assert_not_called()

def test_aiops_orchestrator_run_no_data_drift_no_remediation(mock_dependencies):
    """Test orchestrator does not trigger retraining when no data drift is detected."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "data_drift_detection_enabled": True,
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    # Simulate no data drift detected
    mock_dependencies["DataDriftDetector"].return_value.detect_drift.return_value = (False, {}) 
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["DataDriftDetector"].return_value.detect_drift.assert_called_once()
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()

def test_aiops_orchestrator_run_anomaly_detection_remediation(mock_dependencies):
    """Test orchestrator triggers Redis purge on detected anomalies (e.g., from IsolationForest)."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "anomaly_detection_enabled": True, # Enable anomaly detection for this test
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    # Simulate an anomaly detected by IsolationForest
    mock_dependencies["IsolationForestDetector"].return_value.fit_predict.return_value = np.array([1, -1, 1]).reshape(-1,1) # Needs to be 2D
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["IsolationForestDetector"].return_value.fit_predict.assert_called_once()
    mock_dependencies["RedisRemediator"].return_value.purge_cache.assert_called_once_with(config["redis_cache_pattern"])
    mock_dependencies["DockerRemediator"].return_value.restart_service.assert_not_called()
    mock_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()

def test_aiops_orchestrator_run_multiple_iterations(mock_dependencies):
    """Test the orchestrator runs for multiple iterations."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    orchestrator = AIOpsOrchestrator(config)

    with patch('time.sleep') as mock_sleep:
        orchestrator.run(iterations=3)
        assert mock_sleep.call_count == 2 # Changed assertion to iterations - 1

def test_aiops_orchestrator_init_no_autoencoder_config(mock_dependencies):
    """Test initialization of AIOpsOrchestrator without autoencoder config."""
    config = {
        "check_interval_seconds": 1,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
        # No autoencoder_input_dim
    }
    orchestrator = AIOpsOrchestrator(config)
    assert orchestrator.autoencoder_detector is None
    mock_dependencies["AutoencoderDetector"].assert_not_called()

def test_aiops_orchestrator_run_data_drift_insufficient_data(mock_dependencies):
    """Test data drift detection is skipped when insufficient data is provided."""
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api",
        "data_drift_detection_enabled": True,
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "my_app:*"
    }
    
    # Mock PrometheusClient to return empty data for historical metrics
    mock_dependencies["PrometheusClient"].return_value.get_historical_metric_data_multi.return_value = np.array([]) # Empty array
    
    # Ensure no other anomalies are detected
    mock_dependencies["PrometheusClient"].return_value.get_5xx_error_rate.return_value = 0.01
    mock_dependencies["PrometheusClient"].return_value.get_p95_latency.return_value = 0.1
    mock_dependencies["IsolationForestDetector"].return_value.fit_predict.return_value = np.array([1]).reshape(-1,1) # Needs to be 2D
    mock_dependencies["AutoencoderDetector"].return_value.fit_predict.return_value = np.array([1]).reshape(-1,1) # Needs to be 2D
    
    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run(iterations=1)
    
    mock_dependencies["DataDriftDetector"].return_value.detect_drift.assert_not_called()
    mock_dependencies["OrchestratorLogger"].info.assert_any_call("data_drift_check_skipped", reason="insufficient_data")