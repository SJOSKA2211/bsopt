import asyncio
import time
from typing import Any

import pandas as pd
import structlog

from src.ml.aiops.anomaly_detector import AnomalyDetector
from src.ml.aiops.prometheus_adapter import PrometheusClient
from src.ml.aiops.remediators import BaseRemediator, RemediationPlanner
from src.ml.aiops.health_reporter import HealthReporter
from src.ml.drift import calculate_ks_test, calculate_psi
from src.ml.forecasting.tft_model import PriceTFTModel
from src.shared.observability import (
    post_grafana_annotation,
    setup_logging,
)

from src.ml.aiops.autonomous_guardian import AutonomousGuardian

logger = structlog.get_logger(__name__)

class AutonomousEngine:
    """
    Autonomous orchestrator that combines real-time anomaly detection
    with distribution-based drift analysis, automated remediation,
    and high-level guardian oversight.
    """

    def __init__(
        self,
        detector: AnomalyDetector | None = None,
        remediators: list[BaseRemediator] | None = None,
        config: dict[str, Any] | None = None,
        check_interval: int = 10,
        drift_threshold_psi: float = 0.2,
    ):
        setup_logging()
        self.config = config or {}
        
        # 1. Detection Core
        self.detector = detector or AnomalyDetector(
            engine="transformer", 
            input_dim=self.config.get("transformer_input_dim", 10)
        )
        self.remediators = remediators or RemediationPlanner().remediators.values()
        self.planner = RemediationPlanner(list(self.remediators))
        
        # 2. Thresholds and State
        self.check_interval = self.config.get("check_interval_seconds", check_interval)
        self.drift_threshold_psi = self.config.get("data_drift_psi_threshold", drift_threshold_psi)
        self.is_running = False
        self.reference_data = None
        self.history = []
        self.anomaly_queue_key = "aiops:anomaly_queue"

        # 3. Enhanced Detectors
        self.prometheus_url = self.config.get("prometheus_url")
        self.prometheus_client = (
            PrometheusClient(url=self.prometheus_url) if self.prometheus_url else None
        )
        self.api_service_name = self.config.get("api_service_name", "bsopt-api")

        ae_input_dim = self.config.get("autoencoder_input_dim")
        self.autoencoder_detector = (
            AnomalyDetector(engine="autoencoder", input_dim=ae_input_dim) if ae_input_dim else None
        )

        self.forecaster = (
            PriceTFTModel(config=self.config.get("tft_config"))
            if self.config.get("predictive_scaling_enabled")
            else None
        )
        self.max_history = 1000
        self.last_baseline_update = time.time()
        
        # 4. Guardian Oversight
        self.guardian = AutonomousGuardian(self)

        # 5. Health Reporting
        self.health_reporter = HealthReporter(
            prometheus_url=self.prometheus_url,
            api_service_name=self.api_service_name
        ) if self.prometheus_url else None

    async def _process_redis_anomalies(self):
        """Polls Redis for externally reported anomalies (e.g. from Webhooks)."""
        from src.shared.utils.cache import get_redis
        redis = get_redis()
        if not redis:
            return []

        anomalies = []
        while True:
            data = await redis.lpop(self.anomaly_queue_key)
            if not data:
                break
            try:
                import msgspec
                anomaly = msgspec.json.decode(data)
                anomalies.append(anomaly)
                logger.warning("external_anomaly_received", anomaly=anomaly)
            except Exception as e:
                logger.error("failed_to_decode_external_anomaly", error=str(e))
        return anomalies

    async def run_cycle(self, current_data: pd.DataFrame | None = None):
        """
        Perform one cycle of detection, drift analysis, and remediation.
        """
        logger.info("self_healing_cycle_start")

        try:
            # 1. Gather all anomaly signals
            tasks = [self._process_redis_anomalies()]
            
            if current_data is not None:
                tasks.append(asyncio.to_thread(self.detector.detect, current_data))
                tasks.append(asyncio.to_thread(self._analyze_drift, current_data))
            
            if self.prometheus_client:
                tasks.append(self._detect_system_anomalies())
            
            if self.health_reporter:
                tasks.append(self.health_reporter.get_health_report())

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Extract results safely
            external_anomalies = results[0] if not isinstance(results[0], Exception) else []
            
            idx = 1
            ml_anomalies = []
            if current_data is not None:
                ml_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
                drift_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
            else:
                drift_anomalies = []

            system_anomalies = []
            if self.prometheus_client:
                system_anomalies = results[idx] if not isinstance(results[idx], Exception) else []
                idx += 1
            
            if self.health_reporter:
                health_report = results[idx] if not isinstance(results[idx], Exception) else None
                if health_report:
                    logger.info("health_report_status", status=health_report.status)

            all_anomalies = external_anomalies + ml_anomalies + drift_anomalies + system_anomalies

            if not all_anomalies:
                logger.info("system_health_nominal")
                return

            # 2. Remediation Execution
            for anomaly in all_anomalies:
                actions = self.planner.plan(anomaly)
                for action in actions:
                    if action.can_run():
                        logger.warning("executing_remediation", action=action.name, type=anomaly.get("type"))
                        success = await action.remediate(anomaly)
                        await action.update_last_run()
                        self._record_history(action.name, anomaly, success)
                        
                        if success:
                            post_grafana_annotation(
                                f"Remediated {anomaly.get('type')} via {action.name}",
                                ["aiops", "remediation"]
                            )

        except Exception as e:
            logger.error("self_healing_cycle_error", error=str(e))

    def _record_history(self, action: str, anomaly: dict, success: bool):
        if not hasattr(self, "max_history"):
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
            if not hasattr(self, "last_baseline_update"):
                self.last_baseline_update = time.time()
            logger.info("drift_baseline_initialized")
            return []

        drift_anomalies = []
        numeric_cols = current_data.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            ref_vals = self.reference_data[col].values
            curr_vals = current_data[col].values

            try:
                psi_score = calculate_psi(ref_vals, curr_vals)
                ks_stat, p_val = calculate_ks_test(ref_vals, curr_vals)

                if psi_score > self.drift_threshold_psi:
                    drift_info = {
                        "type": "distribution_drift",
                        "metric": col,
                        "psi_score": float(psi_score),
                        "ks_p_val": float(p_val),
                        "score": float(psi_score),
                    }
                    drift_anomalies.append(drift_info)
                    logger.warning("metric_distribution_drift_detected", **drift_info)
            except Exception:
                pass

        # Periodically update reference data (every 4 hours)
        now = time.time()
        if not hasattr(self, "last_baseline_update"):
            self.last_baseline_update = now
        if now - self.last_baseline_update > 14400:
            self.reference_data = current_data
            self.last_baseline_update = now
            logger.info("drift_baseline_updated", timestamp=now)
        return drift_anomalies

    async def _detect_system_anomalies(self) -> list[dict]:
        """Scans Prometheus for system-level anomalies."""
        if not self.prometheus_client:
            return []

        system_anomalies = []
        try:
            # 1. Error Rate
            error_rate = await self.prometheus_client.get_5xx_error_rate(self.api_service_name)
            if error_rate > self.config.get("error_rate_threshold", 0.05):
                system_anomalies.append(
                    {"type": "high_error_rate", "metric": error_rate, "severity": "high"}
                )

            # 2. Latency
            p95_latency = await self.prometheus_client.get_p95_latency(self.api_service_name)
            if p95_latency > self.config.get("latency_threshold", 0.5):
                system_anomalies.append(
                    {"type": "high_latency", "metric": p95_latency, "severity": "medium"}
                )

            # 3. Predictive (Forecasting)
            if self.forecaster:
                recent_df = await self.prometheus_client.get_metric_range(
                    self.api_service_name, "container_cpu_usage_seconds_total"
                )
                if not recent_df.empty:
                    forecast = self.forecaster.predict(recent_df)
                    if forecast is not None and forecast.max() > 0.8:
                        system_anomalies.append(
                            {"type": "predicted_load_spike", "forecast_max": float(forecast.max())}
                        )

        except Exception as e:
            logger.error("system_anomaly_detection_failed", error=str(e))

        return system_anomalies

    async def start(self, data_source: Any):
        """Start the autonomous self-healing loop and guardian oversight."""
        self.is_running = True
        logger.info("autonomous_engine_started")

        # Start Guardian in a separate task
        asyncio.create_task(self.guardian.monitor_integrity())

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
        """Stop the orchestrator loop and cleanup resources."""
        self.is_running = False
        if hasattr(self.detector, "shutdown"):
            self.detector.shutdown()
        if self.forecaster and hasattr(self.forecaster, "close"):
            self.forecaster.close()
        logger.info("self_healing_orchestrator_stopped")
