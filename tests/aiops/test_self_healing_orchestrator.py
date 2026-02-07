from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.aiops.self_healing_orchestrator import SelfHealingOrchestrator


@patch("src.aiops.self_healing_orchestrator.logger")
class TestSelfHealingOrchestrator:

    def test_init(self, mock_logger):
        """Test initialization of SelfHealingOrchestrator."""
        mock_detector = MagicMock()
        mock_remediator_1 = MagicMock()
        mock_remediator_2 = MagicMock()
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2],
            check_interval=5
        )
        
        assert orchestrator.detector == mock_detector
        assert orchestrator.remediators == [mock_remediator_1, mock_remediator_2]
        assert orchestrator.check_interval == 5
        assert orchestrator.is_running is False
        mock_logger.assert_not_called() # No logger calls in __init__

    @pytest.mark.asyncio
    async def test_run_cycle_no_anomalies(self, mock_logger):
        """Verify that no action is taken when no anomalies are detected."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        
        mock_remediator = MagicMock()
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        await orchestrator.run_cycle(pd.DataFrame({"a": [1]}))
        
        mock_detector.detect.assert_called_once()
        mock_remediator.remediate.assert_not_called()
        mock_logger.info.assert_any_call("self_healing_cycle_start")
        mock_logger.info.assert_any_call("no_anomalies_detected")

    @pytest.mark.asyncio
    async def test_run_cycle_remediation(self, mock_logger):
        """Verify that remediation is triggered when an anomaly is detected."""
        anomaly = {"index": 0, "score": 0.5, "metrics": {"latency": 5.0}, "service": "api", "type": "generic"}
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [anomaly]
        
        mock_remediator_1 = MagicMock()
        mock_remediator_1.supported_types = ["generic"]
        mock_remediator_2 = MagicMock()
        mock_remediator_2.supported_types = ["generic"]
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        await orchestrator.run_cycle(pd.DataFrame({"latency": [5.0]}))
        
        mock_detector.detect.assert_called_once()
        mock_remediator_1.remediate.assert_called_once_with(anomaly)
        mock_remediator_2.remediate.assert_called_once_with(anomaly)
        mock_logger.info.assert_any_call("self_healing_cycle_start")
        mock_logger.warning.assert_called_once_with("anomalies_detected", count=1)

    @pytest.mark.asyncio
    async def test_run_cycle_multiple_anomalies_multiple_remediators(self, mock_logger):
        """Verify that multiple anomalies trigger multiple remediations."""
        anomaly1 = {"index": 0, "service": "api", "type": "generic"}
        anomaly2 = {"index": 1, "service": "worker-ml", "type": "generic"}
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [anomaly1, anomaly2]
        
        mock_remediator_1 = MagicMock()
        mock_remediator_1.supported_types = ["generic"]
        mock_remediator_2 = MagicMock()
        mock_remediator_2.supported_types = ["generic"]
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        await orchestrator.run_cycle(pd.DataFrame({"data": [1,2]}))
        
        mock_detector.detect.assert_called_once()
        assert mock_remediator_1.remediate.call_count == 2
        assert mock_remediator_2.remediate.call_count == 2
        mock_logger.warning.assert_called_once_with("anomalies_detected", count=2)


    @pytest.mark.asyncio
    async def test_run_cycle_exception(self, mock_logger):
        """Verify that exceptions in run_cycle are caught and logged."""
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = Exception("Detection failed")
        
        orchestrator = SelfHealingOrchestrator(detector=mock_detector, remediators=[])
        mock_logger.reset_mock() # Reset logger calls after init
        
        await orchestrator.run_cycle(pd.DataFrame({"a": [1]}))
        
        mock_detector.detect.assert_called_once()
        mock_logger.info.assert_any_call("self_healing_cycle_start")
        mock_logger.error.assert_called_once_with("self_healing_cycle_error", error="Detection failed")

    def test_stop(self, mock_logger):
        """Verify that the orchestrator can be stopped."""
        orchestrator = SelfHealingOrchestrator(MagicMock(), [])
        orchestrator.is_running = True # Manually set to running to test stopping
        mock_logger.reset_mock() # Reset logger calls after init
        
        orchestrator.stop()
        
        assert orchestrator.is_running is False
        mock_logger.info.assert_called_once_with("self_healing_orchestrator_stopped")
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
