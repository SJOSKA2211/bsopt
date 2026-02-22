import asyncio
import time
from typing import Any

import pandas as pd
import structlog

from src.aiops.remediators import BaseRemediator, RemediationPlanner
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector

logger = structlog.get_logger(__name__)


class SelfHealingOrchestrator:
    """
    Autonomous orchestrator that combines real-time anomaly detection
    with distribution-based drift analysis and automated remediation.
    """

    def __init__(
        self,
        detector: TimeSeriesAnomalyDetector,
        remediators: list[BaseRemediator],
        check_interval: int = 10,
        drift_threshold_psi: float = 0.2,
    ):
        self.detector = detector
        self.planner = RemediationPlanner(remediators)
        self.remediators = remediators
        self.check_interval = check_interval
        self.drift_threshold_psi = drift_threshold_psi
        self.is_running = False
        self.reference_data = None
        self.history = []

    async def run_cycle(self, current_data: pd.DataFrame):
        """
        Perform one cycle of detection, drift analysis, and remediation.
        """
        logger.info("self_healing_cycle_start", data_points=len(current_data))

        try:
            # 1. Point Anomaly Detection (Reactive)
            anomalies = await asyncio.to_thread(self.detector.detect, current_data)
            
            # 2. Distribution Drift Analysis (Proactive)
            drift_anomalies = self._analyze_drift(current_data)
            all_anomalies = anomalies + drift_anomalies

            if not all_anomalies:
                logger.info("system_health_nominal")
                return

            logger.warning("anomalies_and_drift_detected", 
                           anomalies=len(anomalies), 
                           drifts=len(drift_anomalies))
            
            # 3. Intelligent Remediation Planning
            for anomaly in all_anomalies:
                actions = self.planner.plan(anomaly)
                if actions:
                    logger.info("executing_remediation_plan", 
                                anomaly_type=anomaly.get("type"), 
                                actions=[a.name for a in actions])
                    for action in actions:
                        if action.can_run():
                            logger.info(
                                "executing_remediation",
                                action=action.name,
                                anomaly=anomaly.get("type"),
                            )
                            # Using gather or individual await? Let's use individual await as in stash
                            success = await action.remediate(anomaly)
                            await action.update_last_run()

                            self._record_history(action.name, anomaly, success)
                        else:
                            logger.debug("remediation_skipped_cooldown", action=action.name)

        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    def _record_history(self, action: str, anomaly: dict, success: bool):
        if not hasattr(self, 'max_history'):
             self.max_history = 1000
        self.history.append(
            {
                "timestamp": time.time(),
                "action": action,
                "anomaly": anomaly.get("type"),
                "success": success,
            }
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _analyze_drift(self, current_data: pd.DataFrame) -> list[dict]:
        """
        Detects shifts in system metric distributions (e.g. baseline latency shift).
        """
        if self.reference_data is None:
            self.reference_data = current_data
            if not hasattr(self, 'last_baseline_update'):
                 self.last_baseline_update = time.time()
            logger.info("drift_baseline_initialized")
            return []

        drift_anomalies = []
        numeric_cols = current_data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            ref_vals = self.reference_data[col].values
            curr_vals = current_data[col].values
            
            try:
                from src.aiops.timeseries_anomaly_detector import calculate_psi, calculate_ks_test
                psi_score = calculate_psi(ref_vals, curr_vals)
                ks_stat, p_val = calculate_ks_test(ref_vals, curr_vals)
                
                if psi_score > self.drift_threshold_psi:
                    drift_info = {
                        "type": "distribution_drift",
                        "metric": col,
                        "psi_score": float(psi_score),
                        "ks_p_val": float(p_val),
                        "score": float(psi_score)
                    }
                    drift_anomalies.append(drift_info)
                    logger.warning("metric_distribution_drift_detected", **drift_info)
            except Exception:
                pass
                
        # Periodically update reference data (every 4 hours)
        now = time.time()
        if not hasattr(self, 'last_baseline_update'):
             self.last_baseline_update = now
        if now - self.last_baseline_update > 14400:
            self.reference_data = current_data
            self.last_baseline_update = now
            logger.info("drift_baseline_updated", timestamp=now)
        return drift_anomalies

    async def start(self, data_source: Any):
        """Start the autonomous self-healing loop."""
        self.is_running = True
        logger.info("self_healing_orchestrator_started")

        while self.is_running:
            try:
                if hasattr(data_source, "get_latest_metrics_async"):
                    data = await data_source.get_latest_metrics_async()
                elif asyncio.iscoroutinefunction(data_source.get_latest_metrics):
                    data = await data_source.get_latest_metrics()
                else:
                    data = await asyncio.to_thread(data_source.get_latest_metrics)

                if isinstance(data, pd.DataFrame):
                    await self.run_cycle(data)
                else:
                    logger.error("invalid_data_format", type=type(data))
            except Exception as e:
                logger.error("loop_iteration_error", error=str(e))

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the orchestrator loop."""
        self.is_running = False
        logger.info("self_healing_orchestrator_stopped")
