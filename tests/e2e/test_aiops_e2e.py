from unittest.mock import ANY, patch

import numpy as np
import pytest

# Import components from src.aiops
from src.aiops.aiops_orchestrator import AIOpsOrchestrator


# Mocks for all components and shared observability functions
@pytest.fixture
def mock_e2e_dependencies():
    with patch(
        "src.aiops.aiops_orchestrator.PrometheusClient"
    ) as MockPrometheusClient, patch(
        "src.aiops.aiops_orchestrator.IsolationForestDetector"
    ) as MockIsolationForestDetector, patch(
        "src.aiops.aiops_orchestrator.AutoencoderDetector"
    ) as MockAutoencoderDetector, patch(
        "src.aiops.aiops_orchestrator.DataDriftDetector"
    ) as MockDataDriftDetector, patch(
        "src.aiops.aiops_orchestrator.DockerRemediator"
    ) as MockDockerRemediator, patch(
        "src.aiops.aiops_orchestrator.MLPipelineTrigger"
    ) as MockMLPipelineTrigger, patch(
        "src.aiops.aiops_orchestrator.RedisRemediator"
    ) as MockRedisRemediator, patch(
        "src.aiops.aiops_orchestrator.setup_logging"
    ) as MockSetupLogging, patch(
        "src.aiops.aiops_orchestrator.logger"
    ) as MockOrchestratorLogger, patch(
        "src.aiops.aiops_orchestrator.post_grafana_annotation"
    ) as MockPostGrafanaAnnotation, patch(
        "src.aiops.aiops_orchestrator.push_metrics"
    ) as MockPushMetrics, patch(
        "src.shared.observability.os.environ.get"
    ) as MockSharedEnvironGet:

        MockSharedEnvironGet.side_effect = lambda key, default=None: {
            "GRAFANA_URL": "http://mock-grafana:3000",
            "PUSHGATEWAY_URL": "http://mock-pushgateway:9091",
        }.get(key, default)

        # Configure PrometheusClient mocks
        mock_prometheus_client_instance = MockPrometheusClient.return_value
        mock_prometheus_client_instance.get_5xx_error_rate.return_value = 0.01
        mock_prometheus_client_instance.get_p95_latency.return_value = 0.1
        mock_prometheus_client_instance.get_historical_metric_data.return_value = (
            np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        )
        mock_prometheus_client_instance.get_historical_metric_data_multi.return_value = np.array(
            [[1.0, 0.5], [2.0, 0.6], [3.0, 0.7]]
        )

        # Configure IsolationForestDetector mocks
        mock_isolation_forest_detector_instance = (
            MockIsolationForestDetector.return_value
        )
        mock_isolation_forest_detector_instance.fit_predict.return_value = np.array(
            [1]
        ).reshape(-1, 1)

        # Configure AutoencoderDetector mocks
        mock_autoencoder_detector_instance = MockAutoencoderDetector.return_value
        mock_autoencoder_detector_instance.fit_predict.return_value = np.array(
            [1]
        ).reshape(-1, 1)

        # Configure DataDriftDetector mocks
        mock_data_drift_detector_instance = MockDataDriftDetector.return_value
        mock_data_drift_detector_instance.detect_drift.return_value = (False, {})
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
            "PostGrafanaAnnotation": MockPostGrafanaAnnotation,
            "PushMetrics": MockPushMetrics,
        }


def test_e2e_api_spike_remediation(mock_e2e_dependencies):
    """
    Simulate an API spike (high 5xx error rate) and verify automated recovery
    (Docker service restart, Grafana annotation, Loki logs).
    """
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api_service_to_restart",
        "error_rate_threshold": 0.05,
        "latency_threshold": 0.5,
        "anomaly_detection_enabled": False,  # Disable other detections for isolation
        "data_drift_detection_enabled": False,
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "test_cache:*",
    }

    # Simulate high 5xx error rate
    mock_e2e_dependencies[
        "PrometheusClient"
    ].return_value.get_5xx_error_rate.return_value = 0.1

    orchestrator = AIOpsOrchestrator(config)

    with patch("time.sleep"):  # Mock sleep to speed up test
        orchestrator.run(iterations=1)  # Run one iteration

    # Verify detection
    mock_e2e_dependencies[
        "PrometheusClient"
    ].return_value.get_5xx_error_rate.assert_called_once()

    # Verify remediation action: Docker restart
    mock_e2e_dependencies[
        "DockerRemediator"
    ].return_value.restart_service.assert_called_once_with(config["api_service_name"])

    # Verify Grafana annotation
    mock_e2e_dependencies["PostGrafanaAnnotation"].assert_called_once_with(
        ANY,  # The message argument
        ["remediation", "api_spike", config["api_service_name"]],  # The tags argument
    )

    # Verify Loki logs (through OrchestratorLogger)
    mock_e2e_dependencies["OrchestratorLogger"].warning.assert_any_call(
        "anomaly_detected",
        type="high_error_rate",
        service=config["api_service_name"],
        value=ANY,
        threshold=ANY,
    )
    mock_e2e_dependencies["OrchestratorLogger"].info.assert_any_call(
        "remediation_action",
        action="restart_service",
        service=config["api_service_name"],
        message=ANY,
    )

    # Verify metrics push
    mock_e2e_dependencies["PushMetrics"].assert_called_once_with(
        job_name="aiops_orchestrator"
    )


def test_e2e_ml_drift_remediation(mock_e2e_dependencies):
    """
    Simulate ML data drift and verify automated recovery
    (ML pipeline retraining, Grafana annotation, Loki logs).
    """
    config = {
        "check_interval_seconds": 0.01,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "api_service_no_drift",
        "anomaly_detection_enabled": False,
        "data_drift_detection_enabled": True,  # Enable data drift detection
        "ml_pipeline_config": {"ticker": "MSFT", "framework": "sklearn"},
        "redis_cache_pattern": "test_cache:*",
    }

    # Simulate data drift detected
    mock_e2e_dependencies[
        "DataDriftDetector"
    ].return_value.detect_drift.return_value = (True, {"drift_types": ["PSI_Drift"]})
    orchestrator = AIOpsOrchestrator(config)

    with patch("time.sleep"):
        orchestrator.run(iterations=1)

    # Verify detection
    mock_e2e_dependencies[
        "DataDriftDetector"
    ].return_value.detect_drift.assert_called_once()

    # Verify remediation action: ML retraining
    mock_e2e_dependencies[
        "MLPipelineTrigger"
    ].return_value.trigger_retraining.assert_called_once()

    # Verify Grafana annotation
    mock_e2e_dependencies["PostGrafanaAnnotation"].assert_called_once_with(
        ANY,  # The message argument
        [
            "remediation",
            "data_drift",
        ],  # The tags argument, without the ML pipeline ticker
    )

    # Verify Loki logs
    mock_e2e_dependencies["OrchestratorLogger"].warning.assert_any_call(
        "anomaly_detected", type="data_drift", info=ANY
    )
    mock_e2e_dependencies["OrchestratorLogger"].info.assert_any_call(
        "remediation_action", action="trigger_ml_retraining", message=ANY
    )

    # Verify metrics push
    mock_e2e_dependencies["PushMetrics"].assert_called_once_with(
        job_name="aiops_orchestrator"
    )
