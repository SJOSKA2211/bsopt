from unittest.mock import MagicMock, patch

import pytest

from src.ml.aiops.aiops_orchestrator import AIOpsOrchestrator

@pytest.fixture
def orchestrator_config():
    return {
        "prometheus_url": "http://mock-prometheus:9090",
        "api_service_name": "bsopt-api",
        "error_rate_threshold": 0.05,
        "latency_threshold": 0.5,
        "anomaly_detection_enabled": True,
        "predictive_scaling_enabled": False,  # Disable heavy TFT for unit test
        "autoencoder_input_dim": None,  # Disable AE for unit test
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
    }

@patch("src.ml.aiops.aiops_orchestrator.PrometheusClient")
@patch("src.ml.aiops.aiops_orchestrator.post_grafana_annotation")
def test_orchestrator_remediates_high_error_rate(
    mock_notify, mock_prometheus_class, orchestrator_config
):
    # Setup mock Prometheus client
    mock_prometheus = mock_prometheus_class.return_value
    mock_prometheus.get_5xx_error_rate.return_value = 0.10  # High error rate!
    mock_prometheus.get_p95_latency.return_value = 0.1
    mock_prometheus.get_historical_metric_data.return_value = []

    # Initialize orchestrator
    orchestrator = AIOpsOrchestrator(orchestrator_config)

    # Mock remediators to avoid side effects
    orchestrator.docker_remediator.restart_service = MagicMock()

    # Run one iteration of detection
    anomalies = orchestrator._detect_anomalies()
    assert "high_error_rate" in anomalies

    # Run remediation
    orchestrator._remediate_anomalies(anomalies)

    # Verify that restart_service was called
    orchestrator.docker_remediator.restart_service.assert_called_with("bsopt-api")
    # Verify notification was sent
    assert mock_notify.called

@patch("src.ml.aiops.aiops_orchestrator.PrometheusClient")
def test_orchestrator_detects_data_drift(mock_prometheus_class, orchestrator_config):
    mock_prometheus = mock_prometheus_class.return_value
    mock_prometheus.get_5xx_error_rate.return_value = 0.01
    mock_prometheus.get_p95_latency.return_value = 0.1

    # Mock data drift detector
    orchestrator = AIOpsOrchestrator(orchestrator_config)
    orchestrator.data_drift_detector.detect_drift = MagicMock(
        return_value=(True, {"p_value": 0.01})
    )
    orchestrator.ml_pipeline_trigger.trigger_retraining = MagicMock()

    # Mock multivariate data
    mock_prometheus.get_historical_metric_data_multi.return_value = [[1, 2], [3, 4]]

    anomalies = orchestrator._detect_anomalies()
    assert "data_drift" in anomalies

    orchestrator._remediate_anomalies(anomalies)
    orchestrator.ml_pipeline_trigger.trigger_retraining.assert_called_once()
