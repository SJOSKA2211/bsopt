import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.aiops.autonomous_engine import AutonomousEngine
from src.ml.aiops.remediators import BaseRemediator


@pytest.fixture
def orchestrator_config():
    return {
        "prometheus_url": "http://mock-prometheus:9090",
        "api_service_name": "bsopt-api",
        "error_rate_threshold": 0.05,
        "latency_threshold": 0.5,
        "anomaly_detection_enabled": True,
        "predictive_scaling_enabled": False,
        "transformer_input_dim": 10,
    }


class MockRemediator(BaseRemediator):
    def __init__(self, name, supported_types):
        super().__init__(name, supported_types)
        self.remediate_mock = MagicMock(return_value=True)

    async def remediate(self, anomaly):
        return self.remediate_mock(anomaly)


@pytest.mark.asyncio
@patch("src.ml.aiops.autonomous_engine.PrometheusClient")
@patch("src.ml.aiops.autonomous_engine.post_grafana_annotation")
async def test_engine_remediates_high_error_rate(
    mock_notify, mock_prometheus_class, orchestrator_config
):
    # Setup mock Prometheus client
    mock_prometheus = mock_prometheus_class.return_value
    mock_prometheus.get_5xx_error_rate.return_value = 0.10  # High error rate!
    mock_prometheus.get_p95_latency.return_value = 0.1

    # Initialize engine with mock remediator
    mock_remediator = MockRemediator("restart_service", ["high_error_rate"])
    engine = AutonomousEngine(config=orchestrator_config, remediators=[mock_remediator])

    # Run one cycle
    await engine.run_cycle()

    # Verify that remediate was called
    mock_remediator.remediate_mock.assert_called_once()
    # Verify notification was sent
    assert mock_notify.called


@pytest.mark.asyncio
@patch("src.ml.aiops.autonomous_engine.PrometheusClient")
async def test_engine_detects_data_drift(mock_prometheus_class, orchestrator_config):
    mock_prometheus = mock_prometheus_class.return_value
    mock_prometheus.get_5xx_error_rate.return_value = 0.01
    mock_prometheus.get_p95_latency.return_value = 0.1

    # Initialize engine
    mock_remediator = MockRemediator("retrain_model", ["data_drift"])
    engine = AutonomousEngine(config=orchestrator_config, remediators=[mock_remediator])

    # Create some dummy data for drift
    ref_data = pd.DataFrame({"val": [1.0] * 100})
    curr_data = pd.DataFrame({"val": [10.0] * 100})  # Significant drift

    # Set reference data
    engine.reference_data = ref_data

    # Run cycle with current data
    await engine.run_cycle(current_data=curr_data)

    # Verify that retrain_model was called
    mock_remediator.remediate_mock.assert_called_once()


@pytest.mark.asyncio
@patch("src.ml.aiops.autonomous_engine.PrometheusClient")
@patch("src.ml.aiops.autonomous_engine.HealthReporter")
async def test_engine_integrates_health_reporter(
    mock_health_reporter_class, mock_prometheus_class, orchestrator_config
):
    mock_health_reporter = mock_health_reporter_class.return_value
    mock_health_reporter.get_health_report.return_value = MagicMock(status="healthy")

    engine = AutonomousEngine(config=orchestrator_config)

    await engine.run_cycle()

    mock_health_reporter.get_health_report.assert_called_once()
