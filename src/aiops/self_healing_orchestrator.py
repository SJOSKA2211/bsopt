import asyncio
import structlog
from typing import Any, List
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector

logger = structlog.get_logger()

class SelfHealingOrchestrator:
    """
    Orchestrator that combines anomaly detection with automated remediation actions.
    Implements a closed-loop control system for infrastructure health.
    Now optimized with asyncio for high-throughput detection.
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

    async def run_cycle(self, current_data: Any):
        """Perform one cycle of detection and remediation (Async)."""
        logger.info("self_healing_cycle_start")
        
        try:
            # 1. Detect anomalies (offload to thread if CPU bound)
            anomalies = await asyncio.to_thread(self.detector.detect, current_data)
            
            if not anomalies:
                logger.info("no_anomalies_detected")
                return

            logger.warning("anomalies_detected", count=len(anomalies))
            
            # 2. Trigger targeted remediation
            tasks = []
            for anomaly in anomalies:
                # 🚀 SOTA: Route anomaly to specific remediators based on type
                a_type = anomaly.get("type", "generic")
                for remediator in self.remediators:
                    # Check if remediator supports this specific anomaly type
                    supported_types = getattr(remediator, "supported_types", ["generic"])
                    if a_type in supported_types:
                        if asyncio.iscoroutinefunction(remediator.remediate):
                            tasks.append(remediator.remediate(anomaly))
                        else:
                            tasks.append(asyncio.to_thread(remediator.remediate, anomaly))
            
            if tasks:
                logger.info("triggering_remediations", task_count=len(tasks))
                await asyncio.gather(*tasks)
                    
        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    async def start(self, data_source: Any):
        """Start the continuous asynchronous self-healing loop."""
        self.is_running = True
        logger.info("self_healing_orchestrator_started")
        
        while self.is_running:
            # Check if metrics retrieval is async
            if hasattr(data_source, "get_latest_metrics_async"):
                 data = await data_source.get_latest_metrics_async()
            elif asyncio.iscoroutinefunction(data_source.get_latest_metrics):
                data = await data_source.get_latest_metrics()
            else:
                data = await asyncio.to_thread(data_source.get_latest_metrics)
                
            await self.run_cycle(data)
            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the orchestrator loop."""
        self.is_running = False
        logger.info("self_healing_orchestrator_stopped")
