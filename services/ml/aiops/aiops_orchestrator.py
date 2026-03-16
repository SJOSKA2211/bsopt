"""
AIOpsOrchestrator — compatibility shim and high-level facade over SelfHealingOrchestrator.

Some test files import `AIOpsOrchestrator` from this module. This implementation
exposes a synchronous API on top of the underlying async engine, integrating
the Prometheus-first anomaly detection pattern.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from services.aiops.anomaly_detector import AnomalyDetector
from services.aiops.docker_remediator import DockerRemediator
from services.aiops.prometheus_adapter import PrometheusClient
from services.aiops.remediators import RemediationPlanner
from services.shared.observability import post_grafana_annotation  # noqa: F401 (re-exported for tests)

logger = structlog.get_logger(__name__)


class AIOpsOrchestrator:
    """
    High-level orchestrator with a synchronous API wrapping our async self-healing core.

    This is the entry-point that older tests and scripts reference. It wires together:
    - PrometheusClient for metric scraping
    - RemediationPlanner for routing anomalies to the correct remediator
    - DockerRemediator for container-level remediation
    - ML-based drift trigger (optional)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        # Infrastructure wiring
        prometheus_url = config.get("prometheus_url", "http://prometheus:9090")
        self.prometheus = PrometheusClient(url=prometheus_url)
        self.api_service_name = config.get("api_service_name", "bsopt-api")
        self.error_rate_threshold = config.get("error_rate_threshold", 0.05)
        self.latency_threshold = config.get("latency_threshold", 0.5)

        # Anomaly detection
        ae_dim = config.get("autoencoder_input_dim")
        self.autoencoder_detector = (
            AnomalyDetector(engine="autoencoder", input_dim=ae_dim) if ae_dim else None
        )

        # Drift detection
        self.data_drift_detector: Any = _NullDriftDetector()

        # Retraining trigger (lazy init to avoid heavy ML imports at test time)
        ml_cfg = config.get("ml_pipeline_config", {})
        self.ml_pipeline_trigger: Any = _NullTrigger()
        if ml_cfg:
            try:
                from services.aiops.ml_pipeline_trigger import MLPipelineTrigger

                self.ml_pipeline_trigger = MLPipelineTrigger(ml_cfg)
            except Exception:  # pragma: no cover
                logger.warning("ml_pipeline_trigger_unavailable")

        # Remediators
        self.docker_remediator = DockerRemediator()
        self.planner = RemediationPlanner()

    # ------------------------------------------------------------------
    # Synchronous Detection API
    # ------------------------------------------------------------------

    def _detect_anomalies(self) -> list[str]:
        """Synchronously poll Prometheus and return a list of anomaly type strings."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_detect())
        finally:
            loop.close()

    async def _async_detect(self) -> list[str]:
        anomalies: list[str] = []
        try:
            error_rate = await asyncio.to_thread(
                self.prometheus.get_5xx_error_rate, self.api_service_name
            )
            if error_rate > self.error_rate_threshold:
                anomalies.append("high_error_rate")
        except Exception as exc:
            logger.warning("error_rate_check_failed", error=str(exc))

        try:
            latency = await asyncio.to_thread(
                self.prometheus.get_p95_latency, self.api_service_name
            )
            if latency > self.latency_threshold:
                anomalies.append("high_latency")
        except Exception as exc:
            logger.warning("latency_check_failed", error=str(exc))

        # Data drift (multivariate)
        try:
            historical = self.prometheus.get_historical_metric_data_multi  # type: ignore
            if callable(historical):
                data = historical()
                if data:
                    drifted, _ = self.data_drift_detector.detect_drift(data, data)
                    if drifted:
                        anomalies.append("data_drift")
        except Exception:
            pass

        return anomalies

    def _remediate_anomalies(self, anomalies: list[str]) -> None:
        """Apply remediations for each detected anomaly type."""
        for anomaly_type in anomalies:
            if anomaly_type == "high_error_rate":
                self.docker_remediator.restart_service(self.api_service_name)
                post_grafana_annotation(
                    f"Restarted {self.api_service_name} due to high error rate",
                    ["aiops", "remediation"],
                )
            elif anomaly_type == "data_drift":
                logger.warning("triggering_mlflow_retrain_run")
                try:
                    # OPTIMIZED: Run asynchronously so we don't block the AIOps event loop
                    import subprocess

                    compose_bin = "docker"
                    subprocess.Popen(
                        [
                            compose_bin,
                            "compose",
                            "exec",
                            "-d",
                            "mlops-worker",
                            "mlflow",
                            "run",
                            ".",
                            "-e",
                            "train_regressor",
                            "-P",
                            f"ticker={self.config.get('ticker', 'AAPL')}",
                            "--experiment-name",
                            f"aiops_retrain_{self.config.get('ticker', 'AAPL')}",
                            "--env-manager",
                            "local",
                        ]
                    )
                except Exception as e:
                    logger.error("failed_to_trigger_mlflow_run", error=str(e))
                    self.ml_pipeline_trigger.trigger_retraining()

    def run_once(self) -> list[str]:
        """Convenience method: detect + remediate in one call."""
        anomalies = self._detect_anomalies()
        if anomalies:
            self._remediate_anomalies(anomalies)
        return anomalies


# ---------------------------------------------------------------------------
# Internal null objects (avoids hard dependencies for lighter test scenarios)
# ---------------------------------------------------------------------------


class _NullDriftDetector:
    """No-op drift detector returned when full setup is unavailable."""

    def detect_drift(self, ref: Any, curr: Any) -> tuple[bool, dict]:  # noqa: ARG002
        return False, {}


class _NullTrigger:
    def trigger_retraining(self) -> None:
        logger.info("ml_retraining_trigger_noop")
