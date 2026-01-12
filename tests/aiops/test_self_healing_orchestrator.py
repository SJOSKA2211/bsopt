import pandas as pd
from unittest.mock import MagicMock
from src.aiops.self_healing_orchestrator import SelfHealingOrchestrator

def test_self_healing_cycle_no_anomalies():
    """Verify that no action is taken when no anomalies are detected."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    
    mock_remediator = MagicMock()
    
    orchestrator = SelfHealingOrchestrator(
        detector=mock_detector,
        remediators=[mock_remediator]
    )
    
    orchestrator.run_cycle(pd.DataFrame({"a": [1]}))
    
    mock_remediator.remediate.assert_not_called()

def test_self_healing_cycle_remediation():
    """Verify that remediation is triggered when an anomaly is detected."""
    anomaly = {"index": 0, "score": -0.5, "metrics": {"latency": 5.0}}
    
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [anomaly]
    
    mock_remediator = MagicMock()
    
    orchestrator = SelfHealingOrchestrator(
        detector=mock_detector,
        remediators=[mock_remediator]
    )
    
    orchestrator.run_cycle(pd.DataFrame({"latency": [5.0]}))
    
    mock_remediator.remediate.assert_called_once_with(anomaly)

def test_self_healing_loop_control():
    """Verify that the orchestrator can be stopped."""
    orchestrator = SelfHealingOrchestrator(MagicMock(), [])
    orchestrator.is_running = True
    orchestrator.stop()
    assert orchestrator.is_running is False
