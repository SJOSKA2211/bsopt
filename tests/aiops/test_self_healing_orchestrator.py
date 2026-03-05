from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.aiops.self_healing_orchestrator import SelfHealingOrchestrator


@patch("src.aiops.self_healing_orchestrator.logger")
@patch("src.aiops.self_healing_orchestrator.setup_logging")
class TestSelfHealingOrchestrator:
    def test_init(self, mock_setup_logging, mock_logger):
        """Test initialization of SelfHealingOrchestrator."""
        mock_detector = MagicMock()
        mock_remediator_1 = MagicMock()
        mock_remediator_2 = MagicMock()
        config = {
            "prometheus_url": "http://localhost:9090",
            "api_service_name": "test-api",
            "check_interval_seconds": 30,
        }

        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2],
            config=config,
        )

        assert orchestrator.detector == mock_detector
        assert orchestrator.remediators == [mock_remediator_1, mock_remediator_2]
        assert orchestrator.check_interval == 30
        assert orchestrator.prometheus_url == "http://localhost:9090"
        assert orchestrator.is_running is False
        mock_setup_logging.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_no_anomalies(self, mock_setup_logging, mock_logger):
        """Verify that no action is taken when no anomalies are detected."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        mock_remediator = MagicMock()

        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector, remediators=[mock_remediator]
        )
        # Mock system anomalies to return empty list
        orchestrator._detect_system_anomalies = AsyncMock(return_value=[])

        await orchestrator.run_cycle(pd.DataFrame({"a": [1]}))

        mock_detector.detect.assert_called_once()
        mock_remediator.remediate.assert_not_called()
        mock_logger.info.assert_any_call("self_healing_cycle_start", data_points=1)
        mock_logger.info.assert_any_call("system_health_nominal")

    @pytest.mark.asyncio
    async def test_run_cycle_remediation(self, mock_setup_logging, mock_logger):
        """Verify that remediation is triggered when an anomaly is detected."""
        anomaly = {
            "type": "generic",
            "score": 0.5,
        }

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [anomaly]

        mock_remediator = MagicMock()
        mock_remediator.name = "GenericRemediator"
        mock_remediator.can_run.return_value = True
        mock_remediator.remediate = AsyncMock(return_value=True)

        # Mock the planner since we are passing remediators directly to orchestrator
        # but the orchestrator uses self.planner = RemediationPlanner(remediators)
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector, remediators=[mock_remediator]
        )
        orchestrator._detect_system_anomalies = AsyncMock(return_value=[])
        orchestrator.planner.plan = MagicMock(return_value=[mock_remediator])

        with patch("src.aiops.self_healing_orchestrator.post_grafana_annotation") as mock_notify:
            await orchestrator.run_cycle(pd.DataFrame({"latency": [5.0]}))

            mock_detector.detect.assert_called_once()
            mock_remediator.remediate.assert_called_once_with(anomaly)
            mock_logger.info.assert_any_call("self_healing_cycle_start", data_points=1)
            mock_notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_system_anomalies(self, mock_setup_logging, mock_logger):
        """Test system anomaly detection via Prometheus client."""
        mock_detector = MagicMock()
        config = {
            "prometheus_url": "http://localhost:9090",
            "api_service_name": "test-api",
            "error_rate_threshold": 0.05,
        }
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector, remediators=[], config=config
        )

        mock_prom = MagicMock()
        mock_prom.get_5xx_error_rate = AsyncMock(return_value=0.1)  # Above threshold
        mock_prom.get_p95_latency = AsyncMock(return_value=0.2)  # Below default threshold 0.5
        orchestrator.prometheus_client = mock_prom

        anomalies = await orchestrator._detect_system_anomalies()

        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "high_error_rate"
        assert anomalies[0]["metric"] == 0.1

    def test_stop(self, mock_setup_logging, mock_logger):
        """Verify that the orchestrator can be stopped."""
        orchestrator = SelfHealingOrchestrator(MagicMock(), [])
        orchestrator.is_running = True
        orchestrator.stop()

        assert orchestrator.is_running is False
        mock_logger.info.assert_any_call("self_healing_orchestrator_stopped")
