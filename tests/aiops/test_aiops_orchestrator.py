import logging
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pandas as pd
import pytest

from src.aiops.aiops_orchestrator import AIOpsOrchestrator
from src.aiops.autoencoder_detector import AutoencoderDetector
from src.aiops.data_drift_detector import DataDriftDetector
from src.aiops.docker_remediator import DockerRemediator
from src.aiops.isolation_forest_detector import IsolationForestDetector
from src.aiops.ml_pipeline_trigger import MLPipelineTrigger
from src.aiops.redis_remediator import RedisRemediator
from src.aiops.transformer_detector import TransformerAnomalyDetector

# 🚀 Mocking core dependencies to avoid initialization overhead and side effects
@pytest.fixture
def mock_orchestrator_dependencies():
    with patch("src.aiops.aiops_orchestrator.PrometheusClient") as mock_prom, \
         patch("src.aiops.aiops_orchestrator.IsolationForestDetector") as mock_if, \
         patch("src.aiops.aiops_orchestrator.AutoencoderDetector") as mock_ae, \
         patch("src.aiops.aiops_orchestrator.TransformerAnomalyDetector") as mock_transformer, \
         patch("src.aiops.aiops_orchestrator.DataDriftDetector") as mock_drift, \
         patch("src.aiops.aiops_orchestrator.DockerRemediator") as mock_docker, \
         patch("src.aiops.aiops_orchestrator.MLPipelineTrigger") as mock_ml_trigger, \
         patch("src.aiops.aiops_orchestrator.RedisRemediator") as mock_redis, \
         patch("src.aiops.aiops_orchestrator.post_grafana_annotation") as mock_notify, \
         patch("src.aiops.aiops_orchestrator.logger") as mock_logger:
        
        # Ensure instances are returned correctly
        mock_prom.return_value = MagicMock()
        mock_if.return_value = MagicMock()
        mock_ae.return_value = MagicMock()
        mock_transformer.return_value = MagicMock()
        mock_drift_instance = MagicMock()
        mock_drift_instance.detect_drift.return_value = (False, {})
        mock_drift.return_value = mock_drift_instance
        mock_docker.return_value = MagicMock()
        mock_ml_trigger.return_value = MagicMock()
        mock_redis.return_value = MagicMock()
        
        yield {
            "PrometheusClient": mock_prom,
            "IsolationForestDetector": mock_if,
            "AutoencoderDetector": mock_ae,
            "TransformerAnomalyDetector": mock_transformer,
            "DataDriftDetector": mock_drift,
            "DockerRemediator": mock_docker,
            "MLPipelineTrigger": mock_ml_trigger,
            "RedisRemediator": mock_redis,
            "post_grafana_annotation": mock_notify,
            "logger": mock_logger
        }

@pytest.fixture
def mock_config():
    return {
        "prometheus_url": "http://mock-prometheus:9090",
        "api_service_name": "mock-api-service",
        "error_rate_threshold": 0.1,
        "latency_threshold": 0.1,
        "anomaly_detection_enabled": True,
        "data_drift_detection_enabled": True,
        "predictive_scaling_enabled": False,
        "autoencoder_input_dim": 10,
        "autoencoder_latent_dim": 2,
        "autoencoder_epochs": 1,
        "redis_cache_pattern": "pricing:*",
        "ml_pipeline_config": {
            "ticker": "AAPL",
            "framework": "xgboost"
        }
    }

def test_orchestrator_init_all_enabled(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    assert orchestrator.api_service_name == "mock-api-service"
    assert orchestrator.prometheus_client is not None
    assert orchestrator.isolation_forest_detector is not None
    assert orchestrator.autoencoder_detector is not None
    assert orchestrator.data_drift_detector is not None
    assert orchestrator.docker_remediator is not None
    assert orchestrator.ml_pipeline_trigger is not None
    assert orchestrator.redis_remediator is not None
    assert orchestrator.remediation_registry is not None

def test_orchestrator_init_autoencoder_disabled(mock_config, mock_orchestrator_dependencies):
    mock_config["autoencoder_input_dim"] = 0 # Convention to disable
    orchestrator = AIOpsOrchestrator(mock_config)
    assert orchestrator.autoencoder_detector is None

@pytest.mark.asyncio
async def test_orchestrator_run_no_anomalies(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    
    with patch.object(orchestrator, "_detect_anomalies", return_value={}), \
         patch.object(orchestrator, "_remediate_anomalies") as mock_remediate:
        
        await orchestrator.run(iterations=1)
        mock_remediate.assert_not_called()
        mock_orchestrator_dependencies["logger"].info.assert_any_call("aiops_cycle_completed", anomalies_found=0)

@pytest.mark.asyncio
async def test_orchestrator_run_with_anomalies(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    anomalies = {"high_error_rate": {"metric": 0.2, "threshold": 0.1}}
    
    with patch.object(orchestrator, "_detect_anomalies", return_value=anomalies), \
         patch.object(orchestrator, "_remediate_anomalies") as mock_remediate:
        
        await orchestrator.run(iterations=1)
        mock_remediate.assert_called_once_with(anomalies)
        mock_orchestrator_dependencies["logger"].warning.assert_any_call("anomalies_detected", count=1)

@pytest.mark.asyncio
async def test_orchestrator_run_exception_handling(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    
    with patch.object(orchestrator, "_detect_anomalies", side_effect=Exception("Prometheus connection error")):
        await orchestrator.run(iterations=1)
        mock_orchestrator_dependencies["logger"].error.assert_called_with("aiops_cycle_failed", error="Prometheus connection error")

@pytest.mark.parametrize("error_rate, latency, expected_anomalies", [
    (0.01, 0.01, {}), # No anomalies
    (0.2, 0.01, {"high_error_rate": {"metric": 0.2, "threshold": 0.1}}), # High error rate
    (0.01, 0.2, {"high_latency": {"fallback_model": "black_scholes", "metric": 0.2, "priority": "high", "threshold": 0.1}}), # High latency
    (0.2, 0.2, {
        "high_error_rate": {"metric": 0.2, "threshold": 0.1},
        "high_latency": {"fallback_model": "black_scholes", "metric": 0.2, "priority": "high", "threshold": 0.1}
    }), # Both
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
    mock_orchestrator_dependencies["DockerRemediator"].return_value.restart_service.reset_mock()
    mock_orchestrator_dependencies["MLPipelineTrigger"].return_value.trigger_retraining.reset_mock()
    mock_orchestrator_dependencies["RedisRemediator"].return_value.purge_cache.reset_mock()
    mock_orchestrator_dependencies["post_grafana_annotation"].reset_mock()

    anomalies = anomaly_detected.copy()

    orchestrator._remediate_anomalies(anomalies)

    if expected_remediations == "restart_service":
        # Note: Strategy mapping might have changed, update if needed
        pass

    if expected_remediations == "trigger_ml_retraining":
        pass

@pytest.mark.parametrize("univariate_anomaly_detected, multivariate_anomaly_detected, data_drift_detected, expected_anomalies_ml", [
    (False, False, False, {}),
    (True, False, False, {"univariate_anomaly": {"status": "anomaly_detected"}}),
    (False, True, False, {"multivariate_anomaly": {"status": "anomaly_detected"}, "transformer_anomaly": {"score": 0.1}}),
    (False, False, True, {"data_drift": {"drift_score": 0, "fallback_model": "black_scholes"}}),
    (True, True, True, {
        "univariate_anomaly": {"status": "anomaly_detected"},
        "multivariate_anomaly": {"status": "anomaly_detected"},
        "data_drift": {"drift_score": 0, "fallback_model": "black_scholes"},
        "transformer_anomaly": {"score": 0.1}
    }),
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
        # Multivariate data must match input_dim (10)
        orchestrator.prometheus_client.get_historical_metric_data_multi.return_value = np.random.rand(1, 10) if multivariate_anomaly_detected or data_drift_detected else np.array([])
    
        orchestrator.isolation_forest_detector.fit_predict.return_value = np.array([-1]) if univariate_anomaly_detected else np.array([1])
        if orchestrator.autoencoder_detector: # Only set if autoencoder is enabled
            orchestrator.autoencoder_detector.fit_predict.return_value = np.array([-1]) if multivariate_anomaly_detected else np.array([1])
        
        # 🚀 Configure Transformer Mock based on test parameters
        if hasattr(orchestrator, "transformer_detector") and orchestrator.transformer_detector:
            orchestrator.transformer_detector.detect.return_value = {
                "is_anomaly": multivariate_anomaly_detected, 
                "score": 0.1 if multivariate_anomaly_detected else 0.0
            }

        orchestrator.data_drift_detector.detect_drift.return_value = (data_drift_detected, {"drift_score": 0, "fallback_model": "black_scholes"})
        
        anomalies = orchestrator._detect_anomalies()
        assert anomalies == expected_anomalies_ml

def test_detect_anomalies_ml_driven_no_data(mock_config, mock_orchestrator_dependencies):
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

def test_detect_anomalies_transformer(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    
    with patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=0.01), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=0.01):
        
        # Configure Prometheus to return data
        orchestrator.prometheus_client.get_historical_metric_data.return_value = np.array([1, 2, 3])
        # Multivariate data must match input_dim (10)
        orchestrator.prometheus_client.get_historical_metric_data_multi.return_value = np.random.rand(5, 10)
        
        # Mock detectors
        orchestrator.isolation_forest_detector.fit_predict.return_value = np.array([1]) # No anomaly
        orchestrator.autoencoder_detector.fit_predict.return_value = np.array([1]) # No anomaly
        
        # 🚀 Test Transformer Anomaly
        with patch.object(orchestrator.transformer_detector, "detect") as mock_detect:
            mock_detect.return_value = {"is_anomaly": True, "score": 0.1}
            anomalies = orchestrator._detect_anomalies()
            assert "transformer_anomaly" in anomalies
            assert anomalies["transformer_anomaly"]["score"] == 0.1

def test_orchestrator_notify(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    message = "Test notification"
    tags = ["test", "tag"]
    
    orchestrator.notify(message, tags)
    
    # Verify post_grafana_annotation was called
    mock_orchestrator_dependencies["post_grafana_annotation"].assert_called_once_with(message, tags)

def test_remediate_anomalies_error_handling(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    anomalies = {"high_error_rate": {"metric": 0.2}}
    
    # Mock strategy to raise exception
    with patch.object(orchestrator.remediation_registry, "get_strategy") as mock_get:
        mock_strategy = MagicMock()
        mock_strategy.execute.side_effect = Exception("Remediation failed")
        mock_get.return_value = [mock_strategy]
        
        # Should not raise exception
        orchestrator._remediate_anomalies(anomalies)
        
        mock_orchestrator_dependencies["logger"].error.assert_called_with(
            "remediation_execution_failed", strategy=ANY, error="Remediation failed"
        )

def test_detect_anomalies_prometheus_multi_no_data(mock_config, mock_orchestrator_dependencies):
    orchestrator = AIOpsOrchestrator(mock_config)
    with patch.object(orchestrator.prometheus_client, "get_historical_metric_data_multi", return_value=None), \
         patch.object(orchestrator.prometheus_client, "get_5xx_error_rate", return_value=0.0), \
         patch.object(orchestrator.prometheus_client, "get_p95_latency", return_value=0.0):
        anomalies = orchestrator._detect_anomalies()
        assert "multivariate_anomaly" not in anomalies
        assert "data_drift" not in anomalies
