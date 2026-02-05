from unittest.mock import MagicMock, call, patch

import pandas as pd

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

    def test_run_cycle_no_anomalies(self, mock_logger):
        """Verify that no action is taken when no anomalies are detected."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        
        mock_remediator = MagicMock()
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        orchestrator.run_cycle(pd.DataFrame({"a": [1]}))
        
        mock_detector.detect.assert_called_once()
        mock_remediator.remediate.assert_not_called()
        mock_logger.info.assert_has_calls([
            call("self_healing_cycle_start"),
            call("no_anomalies_detected")
        ])
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_run_cycle_remediation(self, mock_logger):
        """Verify that remediation is triggered when an anomaly is detected."""
        anomaly = {"index": 0, "score": -0.5, "metrics": {"latency": 5.0}}
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [anomaly]
        
        mock_remediator_1 = MagicMock()
        mock_remediator_2 = MagicMock()
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        orchestrator.run_cycle(pd.DataFrame({"latency": [5.0]}))
        
        mock_detector.detect.assert_called_once()
        mock_remediator_1.remediate.assert_called_once_with(anomaly)
        mock_remediator_2.remediate.assert_called_once_with(anomaly)
        mock_logger.info.assert_called_once_with("self_healing_cycle_start")
        mock_logger.warning.assert_called_once_with("anomalies_detected", count=1)
        mock_logger.error.assert_not_called()

    def test_run_cycle_multiple_anomalies_multiple_remediators(self, mock_logger):
        """Verify that multiple anomalies trigger multiple remediations."""
        anomaly1 = {"index": 0}
        anomaly2 = {"index": 1}
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [anomaly1, anomaly2]
        
        mock_remediator_1 = MagicMock()
        mock_remediator_2 = MagicMock()
        
        orchestrator = SelfHealingOrchestrator(
            detector=mock_detector,
            remediators=[mock_remediator_1, mock_remediator_2]
        )
        mock_logger.reset_mock() # Reset logger calls after init
        
        orchestrator.run_cycle(pd.DataFrame({"data": [1,2]}))
        
        mock_detector.detect.assert_called_once()
        mock_remediator_1.remediate.assert_has_calls([call(anomaly1), call(anomaly2)], any_order=True)
        mock_remediator_2.remediate.assert_has_calls([call(anomaly1), call(anomaly2)], any_order=True)
        assert mock_remediator_1.remediate.call_count == 2
        assert mock_remediator_2.remediate.call_count == 2
        mock_logger.warning.assert_called_once_with("anomalies_detected", count=2)


    def test_run_cycle_exception(self, mock_logger):
        """Verify that exceptions in run_cycle are caught and logged."""
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = Exception("Detection failed")
        
        orchestrator = SelfHealingOrchestrator(detector=mock_detector, remediators=[])
        mock_logger.reset_mock() # Reset logger calls after init
        
        orchestrator.run_cycle(pd.DataFrame({"a": [1]}))
        
        mock_detector.detect.assert_called_once()
        mock_logger.info.assert_called_once_with("self_healing_cycle_start")
        mock_logger.error.assert_called_once_with("self_healing_cycle_error", error="Detection failed")
        mock_logger.warning.assert_not_called()

    @patch("src.aiops.self_healing_orchestrator.time.sleep")
    def test_start_loop_single_iteration(self, mock_sleep, mock_logger):
        """Verify the start loop runs a single iteration and calls run_cycle."""
        mock_detector = MagicMock()
        mock_remediator = MagicMock()
        orchestrator = SelfHealingOrchestrator(detector=mock_detector, remediators=[mock_remediator], check_interval=0.01)
        mock_logger.reset_mock() # Reset logger calls after init
        
        mock_data_source = MagicMock()
        mock_data_source.get_latest_metrics.return_value = pd.DataFrame({"a": [1]})
        
        # Ensure detect returns no anomalies for this specific test
        mock_detector.detect.return_value = [] # <--- Added this line
        
        # Make the loop run once and stop
        original_run_cycle = orchestrator.run_cycle
        def mock_run_cycle_once(data):
            orchestrator.is_running = False # Stop the loop after first cycle
            original_run_cycle(data)
        
        with patch.object(orchestrator, 'run_cycle', side_effect=mock_run_cycle_once) as mock_run_cycle_method:
            
            orchestrator.start(mock_data_source)
            
            assert orchestrator.is_running is False
            mock_data_source.get_latest_metrics.assert_called_once()
            mock_run_cycle_method.assert_called_once_with(mock_data_source.get_latest_metrics.return_value)
            mock_sleep.assert_called_once_with(orchestrator.check_interval) # sleep is called once

            # Assert for both logger calls in the correct order or using assert_has_calls
            mock_logger.info.assert_has_calls([
                call("self_healing_orchestrator_started"),
                call("self_healing_cycle_start"),
                call("no_anomalies_detected")
            ])
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()

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
