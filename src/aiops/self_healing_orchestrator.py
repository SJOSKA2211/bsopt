import asyncio
import time
from typing import Any

import pandas as pd
import structlog

from src.aiops.remediators import BaseRemediator, RemediationPlanner
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector
from src.ml.drift import calculate_ks_test, calculate_psi

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
        drift_threshold_psi: float = 0.2
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
                    
                    tasks = [a.remediate(anomaly) for a in actions]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Log results and potentially trigger further verification
                    for action, result in zip(actions, results):
                        if isinstance(result, Exception):
                            logger.error("remediation_action_failed", action=action.name, error=str(result))
                        else:
                            logger.info("remediation_action_completed", action=action.name, success=result)
                    
        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    def _analyze_drift(self, current_data: pd.DataFrame) -> list[dict]:
        """
        Detects shifts in system metric distributions (e.g. baseline latency shift).
        """
        if self.reference_data is None:
            self.reference_data = current_data
            logger.info("drift_baseline_initialized")
            return []

        drift_anomalies = []
        numeric_cols = current_data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            ref_vals = self.reference_data[col].values
            curr_vals = current_data[col].values
            
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
                
        # Periodically update reference data to adapt to intentional shifts
        if time.time() % 3600 < self.check_interval:
            self.reference_data = current_data
            logger.info("drift_baseline_updated")
            
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
