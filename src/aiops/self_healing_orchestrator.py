import time
import structlog
from typing import Any, List
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector

logger = structlog.get_logger()

class SelfHealingOrchestrator:
    """
    Orchestrator that combines anomaly detection with automated remediation actions.
    Implements a closed-loop control system for infrastructure health.
    """
    def __init__(
        self, 
        detector: TimeSeriesAnomalyDetector,
        remediators: List[Any],
        check_interval: int = 10
    ):
        self.detector = detector
        self.remediators = remediators
        self.check_interval = check_interval
        self.is_running = False

    def run_cycle(self, current_data: Any):
        """Perform one cycle of detection and remediation."""
        logger.info("self_healing_cycle_start")
        
        try:
            # 1. Detect anomalies
            anomalies = self.detector.detect(current_data)
            
            if not anomalies:
                logger.info("no_anomalies_detected")
                return

            logger.warning("anomalies_detected", count=len(anomalies))
            
            # 2. Trigger remediation for each anomaly
            for anomaly in anomalies:
                for remediator in self.remediators:
                    remediator.remediate(anomaly)
                    
        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    def start(self, data_source: Any):
        """Start the continuous self-healing loop."""
        self.is_running = True
        logger.info("self_healing_orchestrator_started")
        
        while self.is_running:
            data = data_source.get_latest_metrics()
            self.run_cycle(data)
            time.sleep(self.check_interval)

    def stop(self):
        """Stop the orchestrator loop."""
        self.is_running = False
        logger.info("self_healing_orchestrator_stopped")
