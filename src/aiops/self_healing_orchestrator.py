import asyncio
import time
from typing import Any

import pandas as pd
import structlog

from src.aiops.anomaly_detector import AnomalyDetector
from src.aiops.prometheus_adapter import PrometheusClient
from src.aiops.remediators import BaseRemediator, RemediationPlanner
from src.ml.drift import calculate_ks_test, calculate_psi
from src.ml.forecasting.tft_model import PriceTFTModel
from src.shared.observability import (
    post_grafana_annotation,
    setup_logging,
)

logger = structlog.get_logger(__name__)


class SelfHealingOrchestrator:
    """
    Autonomous orchestrator that combines real-time anomaly detection
    with distribution-based drift analysis and automated remediation.
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        remediators: list[BaseRemediator],
        config: dict[str, Any] | None = None,
        check_interval: int = 10,
        drift_threshold_psi: float = 0.2,
    ):
        setup_logging()
        self.detector = detector
        self.planner = RemediationPlanner(remediators)
        self.remediators = remediators
        self.config = config or {}
        self.check_interval = self.config.get("check_interval_seconds", check_interval)
        self.drift_threshold_psi = self.config.get("data_drift_psi_threshold", drift_threshold_psi)
        self.is_running = False
        self.reference_data = None
        self.history = []

        # Prometheus & Advanced Detectors
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

    async def run_cycle(self, current_data: pd.DataFrame):
        """
        Perform one cycle of detection, drift analysis, and remediation (Optimized).
        """
        logger.info("self_healing_cycle_start", data_points=len(current_data))

        try:
            # 1. RUN ALL DETECTION CONCURRENTLY ( Concurrency Fusion)
            # - Reactive: Point anomalies (ML)
            # - Proactive: System anomalies (Prometheus)
            # - Baseline: Distribution drift (Statistical)

            tasks = [
                asyncio.to_thread(self.detector.detect, current_data),
                self._detect_system_anomalies(),
                asyncio.to_thread(self._analyze_drift, current_data),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            anomalies = results[0] if not isinstance(results[0], Exception) else []
            system_anomalies = results[1] if not isinstance(results[1], Exception) else []
            drift_anomalies = results[2] if not isinstance(results[2], Exception) else []

            if isinstance(results[0], Exception):
                logger.error("anomaly_detector_failed", error=str(results[0]))
            if isinstance(results[2], Exception):
                logger.error("drift_analysis_failed", error=str(results[2]))

            all_anomalies = anomalies + system_anomalies + drift_anomalies

            if not all_anomalies:
                logger.info("system_health_nominal")
                return

            logger.warning(
                "anomalies_detected",
                point=len(anomalies),
                system=len(system_anomalies),
                drift=len(drift_anomalies),
            )

            # 4. Intelligent Remediation Planning
            # Remediations are kept sequential to avoid state corruption/race conditions
            # in the infrastructure, but the planning is O(1).
            for anomaly in all_anomalies:
                actions = self.planner.plan(anomaly)
                if actions:
                    logger.info(
                        "executing_remediation_plan",
                        anomaly_type=anomaly.get("type"),
                        actions=[a.name for a in actions],
                    )
                    for action in actions:
                        if action.can_run():
                            logger.info(
                                "executing_remediation",
                                action=action.name,
                                anomaly=anomaly.get("type"),
                            )
                            success = await action.remediate(anomaly)
                            await action.update_last_run()

                            self._record_history(action.name, anomaly, success)
                            if success:
                                post_grafana_annotation(
                                    f"Remediated {anomaly.get('type')} via {action.name}",
                                    ["aiops", "remediation"],
                                )
                        else:
                            logger.debug("remediation_skipped_cooldown", action=action.name)

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
