from unittest.mock import ANY, patch

import numpy as np
import pytest


@pytest.fixture
def mock_config():
    return {
        "check_interval_seconds": 1,
        "prometheus_url": "http://mock-prometheus:9090",
        "api_service_name": "mock-api-service",
        "error_rate_threshold": 0.1,
        "latency_threshold": 0.1,
        "isolation_forest_contamination": 0.05,
        "autoencoder_input_dim": 10,
        "autoencoder_latent_dim": 5,
        "autoencoder_epochs": 1,
        "autoencoder_threshold_multiplier": 2.5,
        "data_drift_psi_threshold": 0.15,
        "data_drift_ks_threshold": 0.01,
        "anomaly_detection_enabled": True, # Default to True for base config
        "data_drift_detection_enabled": True, # Default to True for base config
        "ml_pipeline_config": {"key": "value"},
        "redis_cache_pattern": "test_cache:*"
    }

# AIOpsOrchestrator imports these directly, so we patch them where they are used in the orchestrator module.
# instead of their original definition module.
@pytest.fixture
def mock_orchestrator_dependencies():
    with patch("src.aiops.aiops_orchestrator.PrometheusClient") as mock_prometheus_client_cls, \
         patch("src.aiops.aiops_orchestrator.IsolationForestDetector") as mock_isolation_forest_cls, \
         patch("src.aiops.aiops_orchestrator.AutoencoderDetector") as mock_autoencoder_cls, \
         patch("src.aiops.aiops_orchestrator.DataDriftDetector") as mock_data_drift_cls, \
         patch("src.aiops.aiops_orchestrator.DockerRemediator") as mock_docker_remediator_cls, \
         patch("src.aiops.aiops_orchestrator.MLPipelineTrigger") as mock_ml_pipeline_trigger_cls, \
         patch("src.aiops.aiops_orchestrator.RedisRemediator") as mock_redis_remediator_cls, \
         patch("src.aiops.aiops_orchestrator.setup_logging") as mock_setup_logging, \
         patch("src.aiops.aiops_orchestrator.push_metrics") as mock_push_metrics, \
         patch("src.aiops.aiops_orchestrator.post_grafana_annotation") as mock_post_grafana_annotation, \
         patch("src.aiops.aiops_orchestrator.logger") as mock_orchestrator_logger: # Patch the logger directly
    
        # Configure PrometheusClient instance methods to return numpy arrays
        mock_prometheus_client_cls.return_value.get_historical_metric_data.return_value = np.array([1, 2, 3])
        mock_prometheus_client_cls.return_value.get_historical_metric_data_multi.return_value = np.array([[1, 2], [3, 4]])
        mock_prometheus_client_cls.return_value.get_5xx_error_rate.return_value = 0.0 # Default value
        mock_prometheus_client_cls.return_value.get_p95_latency.return_value = 0.0 # Default value


        # Configure IsolationForestDetector instance methods
        mock_isolation_forest_cls.return_value.fit_predict.return_value = np.array([-1]) # Default anomaly

        # Configure AutoencoderDetector instance methods
        mock_autoencoder_cls.return_value.fit_predict.return_value = np.array([-1]) # Default anomaly

        # Configure DataDriftDetector instance methods
        mock_data_drift_cls.return_value.detect_drift.return_value = (True, {"drift_type": "KS"}) # Default drift

        yield {
            "PrometheusClient": mock_prometheus_client_cls,
            "IsolationForestDetector": mock_isolation_forest_cls,
            "AutoencoderDetector": mock_autoencoder_cls,
            "DataDriftDetector": mock_data_drift_cls,
            "DockerRemediator": mock_docker_remediator_cls,
            "MLPipelineTrigger": mock_ml_pipeline_trigger_cls,
            "RedisRemediator": mock_redis_remediator_cls,
            "setup_logging": mock_setup_logging,
            "push_metrics": mock_push_metrics,
            "post_grafana_annotation": mock_post_grafana_annotation,
            "logger": mock_orchestrator_logger, # Yield the patched logger
        }

from src.aiops.aiops_orchestrator import AIOpsOrchestrator


def test_orchestrator_init_all_enabled(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    assert orchestrator.config == mock_config
    assert orchestrator.check_interval_seconds == mock_config["check_interval_seconds"]
    assert orchestrator.prometheus_client is not None
    assert orchestrator.isolation_forest_detector is not None
    assert orchestrator.autoencoder_detector is not None
    assert orchestrator.data_drift_detector is not None
    assert orchestrator.docker_remediator is not None
    assert orchestrator.ml_pipeline_trigger is not None
    assert orchestrator.redis_remediator is not None
    
    # Verify that the mock constructors were called
    mock_orchestrator_dependencies["PrometheusClient"].assert_called_once_with(url=mock_config["prometheus_url"])
    mock_orchestrator_dependencies["IsolationForestDetector"].assert_called_once_with(contamination=mock_config["isolation_forest_contamination"])
    mock_orchestrator_dependencies["AutoencoderDetector"].assert_called_once_with(
        input_dim=mock_config["autoencoder_input_dim"],
        latent_dim=mock_config["autoencoder_latent_dim"],
        epochs=mock_config["autoencoder_epochs"],
        threshold_multiplier=mock_config["autoencoder_threshold_multiplier"],
        verbose=False
    )
    mock_orchestrator_dependencies["DataDriftDetector"].assert_called_once_with(
        psi_threshold=mock_config["data_drift_psi_threshold"],
        ks_threshold=mock_config["data_drift_ks_threshold"]
    )
    mock_orchestrator_dependencies["DockerRemediator"].assert_called_once()
    mock_orchestrator_dependencies["MLPipelineTrigger"].assert_called_once_with(config=mock_config["ml_pipeline_config"])
    mock_orchestrator_dependencies["RedisRemediator"].assert_called_once()
    # Use ANY for config dict comparison as it might be a copy or slightly different representation in structlog
    mock_orchestrator_dependencies["logger"].info.assert_called_with("aiops_orchestrator_init", status="success", config=ANY)

def test_orchestrator_init_autoencoder_disabled(mock_config, mock_orchestrator_dependencies):
    config_no_autoencoder = mock_config.copy()
    config_no_autoencoder["autoencoder_input_dim"] = None # Disable autoencoder
    
    orchestrator = AIOpsOrchestrator(config_no_autoencoder)
    assert orchestrator.autoencoder_detector is None

    # Verify AutoencoderDetector was NOT called
    mock_orchestrator_dependencies["AutoencoderDetector"].assert_not_called()

@patch("src.aiops.aiops_orchestrator.AIOpsOrchestrator._detect_anomalies")
@patch("src.aiops.aiops_orchestrator.AIOpsOrchestrator._remediate_anomalies")
@patch("time.sleep")
def test_orchestrator_run_no_anomalies(mock_sleep, mock_remediate, mock_detect, mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    mock_detect.return_value = {} # No anomalies
    orchestrator.run(iterations=2) # Run for 2 iterations to ensure sleep is called
    
    assert mock_detect.call_count == 2
    mock_remediate.assert_not_called()
    mock_sleep.assert_called_with(orchestrator.check_interval_seconds)
    mock_orchestrator_dependencies["push_metrics"].assert_called_with(job_name="aiops_orchestrator")
    assert mock_orchestrator_dependencies["push_metrics"].call_count == 2 # Called twice in the loop

@patch("src.aiops.aiops_orchestrator.AIOpsOrchestrator._detect_anomalies")
@patch("src.aiops.aiops_orchestrator.AIOpsOrchestrator._remediate_anomalies")
@patch("time.sleep")
def test_orchestrator_run_with_anomalies(mock_sleep, mock_remediate, mock_detect, mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    mock_detect.return_value = {"high_error_rate": True}
    orchestrator.run(iterations=2) # Run for 2 iterations to ensure sleep is called
    
    assert mock_detect.call_count == 2 # Called twice
    assert mock_remediate.call_count == 2 # Remediation called twice
    mock_remediate.assert_called_with({"high_error_rate": True}) # Check arguments for any call
    mock_sleep.assert_called_with(orchestrator.check_interval_seconds)
    assert mock_orchestrator_dependencies["push_metrics"].call_count == 2 # Called twice

@patch("src.aiops.aiops_orchestrator.AIOpsOrchestrator._detect_anomalies")
@patch("time.sleep")
def test_orchestrator_run_exception_handling(mock_sleep, mock_detect, mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    mock_detect.side_effect = Exception("Test error")
    orchestrator.run(iterations=1)
    
    mock_detect.assert_called_once()
    mock_orchestrator_dependencies["logger"].error.assert_called_with("aiops_orchestrator_loop_error", error=ANY) # Use ANY for error message
    mock_orchestrator_dependencies["push_metrics"].assert_called_once_with(job_name="aiops_orchestrator")


@pytest.mark.parametrize("error_rate, latency, expected_anomalies", [
    (0.01, 0.01, {}), # No anomalies
    (0.2, 0.01, {"high_error_rate": True}), # High error rate
    (0.01, 0.2, {"high_latency": True}), # High latency
    (0.2, 0.2, {"high_error_rate": True, "high_latency": True}), # Both
])
def test_detect_anomalies_prometheus_metrics(mock_config, error_rate, latency, expected_anomalies, mock_orchestrator_dependencies):
    # Create a copy of mock_config and disable ML detections for this specific test
    config_prometheus_only = mock_config.copy()
    config_prometheus_only["anomaly_detection_enabled"] = False
    config_prometheus_only["data_drift_detection_enabled"] = False
    orchestrator = AIOpsOrchestrator(config_prometheus_only)

    with patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=error_rate), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=latency):
        
        anomalies = orchestrator._detect_anomalies()
        assert anomalies == expected_anomalies
        orchestrator.prometheus_client.get_5xx_error_rate.assert_called_once_with(service=orchestrator.api_service_name)
        orchestrator.prometheus_client.get_p95_latency.assert_called_once_with(service=orchestrator.api_service_name)
        
        # Assert that ML-driven detectors were NOT called
        orchestrator.isolation_forest_detector.fit_predict.assert_not_called()
        if orchestrator.autoencoder_detector:
            orchestrator.autoencoder_detector.fit_predict.assert_not_called()
        orchestrator.data_drift_detector.detect_drift.assert_not_called()

@pytest.mark.parametrize("anomaly_detected, drift_detected, expected_remediations", [
    ({"high_error_rate": True}, False, "restart_service"),
    ({"high_latency": True}, False, "restart_service"),
    ({"data_drift": True}, True, "trigger_ml_retraining"),
    ({"univariate_anomaly": True}, False, "purge_redis_cache"),
    ({"multivariate_anomaly": True}, False, "purge_redis_cache"),
])
def test_remediate_anomalies(mock_config, anomaly_detected, drift_detected, expected_remediations, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    # Reset mocks for each test run to ensure clean state
    # We patch at the module level for the classes, so we need to access their return_value (instance) mocks
    mock_orchestrator_dependencies["DockerRemediator"].return_value.restart_service.reset_mock()
    mock_orchestrator_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.reset_mock()
    mock_orchestrator_dependencies["RedisRemediator"].return_value.purge_cache.reset_mock()
    mock_orchestrator_dependencies["post_grafana_annotation"].reset_mock()

    anomalies = anomaly_detected.copy() # Make a copy to avoid modifying fixture for other tests

    orchestrator._remediate_anomalies(anomalies)

    if expected_remediations == "restart_service":
        mock_orchestrator_dependencies["DockerRemediator"].return_value.restart_service.assert_called_once_with(orchestrator.api_service_name)
        mock_orchestrator_dependencies["post_grafana_annotation"].assert_called_once()
    else:
        mock_orchestrator_dependencies["DockerRemediator"].return_value.restart_service.assert_not_called()

    if expected_remediations == "trigger_ml_retraining":
        mock_orchestrator_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_called_once()
        mock_orchestrator_dependencies["post_grafana_annotation"].assert_called_once()
    else:
        mock_orchestrator_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.assert_not_called()

    if expected_remediations == "purge_redis_cache":
        mock_orchestrator_dependencies["RedisRemediator"].return_value.purge_cache.assert_called_once_with(orchestrator.redis_cache_pattern)
        mock_orchestrator_dependencies["post_grafana_annotation"].assert_called_once()
    else:
        mock_orchestrator_dependencies["RedisRemediator"].return_value.purge_cache.assert_not_called()

    if expected_remediations not in ["restart_service", "trigger_ml_retraining", "purge_redis_cache"]:
        mock_orchestrator_dependencies["post_grafana_annotation"].assert_not_called()


@pytest.mark.parametrize("univariate_anomaly_detected, multivariate_anomaly_detected, data_drift_detected, expected_anomalies_ml", [
    (False, False, False, {}),
    (True, False, False, {"univariate_anomaly": True}),
    (False, True, False, {"multivariate_anomaly": True}),
    (False, False, True, {"data_drift": True}),
    (True, True, True, {"univariate_anomaly": True, "multivariate_anomaly": True, "data_drift": True}),
])
def test_detect_anomalies_ml_driven(mock_config, univariate_anomaly_detected, multivariate_anomaly_detected, data_drift_detected, expected_anomalies_ml, mock_orchestrator_dependencies):
    # Create a specific config for this parameterization
    current_mock_config = mock_config.copy()
    current_mock_config["anomaly_detection_enabled"] = univariate_anomaly_detected or multivariate_anomaly_detected
    current_mock_config["data_drift_detection_enabled"] = data_drift_detected
    orchestrator = AIOpsOrchestrator(current_mock_config)

    with patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=0.01), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=0.01):
        
        # Explicitly control Prometheus client return values based on expected calls
        orchestrator.prometheus_client.get_historical_metric_data.return_value = np.array([1]) if univariate_anomaly_detected else np.array([])
        orchestrator.prometheus_client.get_historical_metric_data_multi.return_value = np.array([[1, 2]]) if multivariate_anomaly_detected or data_drift_detected else np.array([])
        
        orchestrator.isolation_forest_detector.fit_predict.return_value = np.array([-1]) if univariate_anomaly_detected else np.array([1])
        if orchestrator.autoencoder_detector: # Only set if autoencoder is enabled
            orchestrator.autoencoder_detector.fit_predict.return_value = np.array([-1]) if multivariate_anomaly_detected else np.array([1])
        orchestrator.data_drift_detector.detect_drift.return_value = (data_drift_detected, {"drift_type": "KS"})
        
        anomalies = orchestrator._detect_anomalies()
        assert anomalies == expected_anomalies_ml

        if univariate_anomaly_detected:
            orchestrator.isolation_forest_detector.fit_predict.assert_called_once_with(ANY)
        else:
            orchestrator.isolation_forest_detector.fit_predict.assert_not_called()

        if multivariate_anomaly_detected and orchestrator.autoencoder_detector:
            orchestrator.autoencoder_detector.fit_predict.assert_called_once_with(ANY)
        else:
            if orchestrator.autoencoder_detector:
                orchestrator.autoencoder_detector.fit_predict.assert_not_called()

        if data_drift_detected:
            orchestrator.data_drift_detector.detect_drift.assert_called_once_with(ANY, ANY)
        else:
            orchestrator.data_drift_detector.detect_drift.assert_not_called()


def test_detect_anomalies_ml_driven_no_data(mock_config, mock_orchestrator_dependencies):
    # Create a specific config for this test
    current_mock_config = mock_config.copy()
    current_mock_config["anomaly_detection_enabled"] = True
    current_mock_config["data_drift_detection_enabled"] = True
    orchestrator = AIOpsOrchestrator(current_mock_config)

    with patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=0.01), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=0.01):
        
        # Directly set return values for no data scenario
        orchestrator.prometheus_client.get_historical_metric_data.return_value = np.array([])
        orchestrator.prometheus_client.get_historical_metric_data_multi.return_value = np.array([])
        
        anomalies = orchestrator._detect_anomalies()
        assert anomalies == {} # No anomalies detected if no data
        orchestrator.isolation_forest_detector.fit_predict.assert_not_called()
        if orchestrator.autoencoder_detector:
            orchestrator.autoencoder_detector.fit_predict.assert_not_called()
        orchestrator.data_drift_detector.detect_drift.assert_not_called()
        mock_orchestrator_dependencies["logger"].info.assert_called_with("data_drift_check_skipped", reason=ANY)

def test_detect_anomalies_disabled(mock_config, mock_orchestrator_dependencies):
    # Create a copy of mock_config and disable ML detections for this specific test
    config_ml_disabled = mock_config.copy()
    config_ml_disabled["anomaly_detection_enabled"] = False
    config_ml_disabled["data_drift_detection_enabled"] = False
    orchestrator = AIOpsOrchestrator(config_ml_disabled)

    with patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=0.01), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=0.01):
        
        # Ensure Prometheus client methods return something with .size
        orchestrator.prometheus_client.get_historical_metric_data.return_value = np.array([])
        orchestrator.prometheus_client.get_historical_metric_data_multi.return_value = np.array([])

        orchestrator.isolation_forest_detector.fit_predict.reset_mock() # Reset to ensure no lingering calls
        if orchestrator.autoencoder_detector:
            orchestrator.autoencoder_detector.fit_predict.reset_mock()
        orchestrator.data_drift_detector.detect_drift.reset_mock()
        
        anomalies = orchestrator._detect_anomalies()
        assert anomalies == {}

        orchestrator.isolation_forest_detector.fit_predict.assert_not_called()
        if orchestrator.autoencoder_detector:
            orchestrator.autoencoder_detector.fit_predict.assert_not_called()
        orchestrator.data_drift_detector.detect_drift.assert_not_called()
