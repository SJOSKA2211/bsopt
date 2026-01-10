import pytest
from unittest.mock import MagicMock, patch
import os
import time

from src.aiops.aiops_orchestrator import AIOpsOrchestrator

import numpy as np

# Mock DockerClient to avoid actual Docker interactions during tests
@pytest.fixture(autouse=True)
def mock_docker_client():
    with patch('src.aiops.docker_remediator.docker.from_env') as mock_from_env:
        mock_container = MagicMock()
        mock_container.name = "mock_service"
        mock_docker_client_instance = MagicMock()
        mock_docker_client_instance.containers.get.return_value = mock_container
        mock_from_env.return_value = mock_docker_client_instance
        yield mock_docker_client_instance

@pytest.fixture
def base_config():
    return {
        "check_interval_seconds": 0.1,
        "prometheus_url": "http://localhost:9090",
        "api_service_name": "test_api_service",
        "error_rate_threshold": 0.05,
        "latency_threshold": 0.5,
        "ml_pipeline_config": {"ticker": "AAPL", "framework": "xgboost"},
        "redis_cache_pattern": "test_cache:*",
        "anomaly_detection_enabled": False,
        "data_drift_detection_enabled": False,
    }

def test_api_spike_e2e_remediation(base_config, mock_docker_client):
    """
    Simulate an API spike (high error rate) and verify that DockerRemediator.restart_service is called.
    This is an E2E test focusing on the interaction between PrometheusClient (mocked),
    AIOpsOrchestrator, and DockerRemediator (mocked).
    """
    with patch('src.aiops.aiops_orchestrator.PrometheusClient') as MockPrometheusClient, \
         patch('src.aiops.aiops_orchestrator.logger') as MockOrchestratorLogger, \
         patch('src.aiops.aiops_orchestrator.RedisRemediator') as MockRedisRemediator, \
         patch('src.aiops.aiops_orchestrator.DockerRemediator') as MockDockerRemediator, \
         patch('src.aiops.aiops_orchestrator.MLPipelineTrigger') as MockMLPipelineTrigger:

        mock_prometheus_client_instance = MockPrometheusClient.return_value
        # Simulate high error rate
        mock_prometheus_client_instance.get_5xx_error_rate.return_value = 0.1
        mock_prometheus_client_instance.get_p95_latency.return_value = 0.1 # Low latency

        orchestrator = AIOpsOrchestrator(base_config)
        
        orchestrator.run(iterations=1)

        # Assert that get_5xx_error_rate was called
        mock_prometheus_client_instance.get_5xx_error_rate.assert_called_with(service=base_config["api_service_name"])
        
        # Assert that the DockerRemediator's restart_service was called
        orchestrator.docker_remediator.restart_service.assert_called_once_with(base_config["api_service_name"])
        MockOrchestratorLogger.info.assert_any_call(
            "remediation_action", action="restart_service", service=base_config["api_service_name"]
        )

# @patch('src.aiops.aiops_orchestrator.DataDriftDetector')
# def test_ml_drift_e2e_remediation(MockDataDriftDetector, base_config, mock_docker_client):
#     """
#     Simulate ML data drift and verify that MLPipelineTrigger.trigger_retraining is called.
#     This is an E2E test focusing on the interaction between DataDriftDetector (mocked),
#     AIOpsOrchestrator, and MLPipelineTrigger (mocked).
#     """
#     with patch('src.aiops.aiops_orchestrator.PrometheusClient') as MockPrometheusClient, \
#          patch('src.aiops.aiops_orchestrator.logger') as MockOrchestratorLogger, \
#          patch('src.aiops.aiops_orchestrator.RedisRemediator') as MockRedisRemediator, \
#          patch('src.aiops.aiops_orchestrator.DockerRemediator') as MockDockerRemediator, \
#          patch('src.aiops.aiops_orchestrator.MLPipelineTrigger') as MockMLPipelineTrigger:

#         mock_data_drift_detector_instance = MockDataDriftDetector.return_value
#         mock_data_drift_detector_instance.detect_drift.return_value = (True, {"drift_types": ["PSI_Drift"]})
        
#         config = base_config.copy()
#         config["data_drift_detection_enabled"] = True

#         orchestrator = AIOpsOrchestrator(config)
#         assert isinstance(orchestrator.data_drift_detector, MagicMock)
#         mock_data_drift_detector_instance = orchestrator.data_drift_detector
#         mock_data_drift_detector_instance.detect_drift.return_value = (True, {"drift_types": ["PSI_Drift"]})
#         orchestrator.run(iterations=1)

#         mock_data_drift_detector_instance.detect_drift.assert_called_once()
#         mock_prometheus_client_instance.get_historical_metric_data_multi.assert_called_with(
#             queries=[f'metric_feature_1{{service="{config["api_service_name"]}"}}[5m]'],
#             duration_seconds=300
#         )
#         assert mock_prometheus_client_instance.get_historical_metric_data_multi.call_count == 2
#         MockMLPipelineTrigger.return_value.trigger_retraining.assert_called_once()
#         MockOrchestratorLogger.info.assert_any_call(
#             "remediation_action", action="trigger_ml_retraining"
#         )
#         MockDockerRemediator.return_value.restart_service.assert_not_called()
#         MockRedisRemediator.return_value.purge_cache.assert_not_called()


