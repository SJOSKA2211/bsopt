import asyncio
import time
from typing import Any

import structlog

from src.aiops.autoencoder_detector import AutoencoderDetector
from src.aiops.data_drift_detector import DataDriftDetector
from src.aiops.docker_remediator import DockerRemediator
from src.aiops.isolation_forest_detector import IsolationForestDetector
from src.aiops.ml_pipeline_trigger import MLPipelineTrigger
from src.aiops.prometheus_adapter import PrometheusClient
from src.aiops.redis_remediator import RedisRemediator
from src.aiops.remediation_strategies import (
    AutonomousScalerStrategy,
    ModelSwitchStrategy,
    PurgeCacheStrategy,
    RemediationRegistry,
    RestartServiceStrategy,
    RetrainModelStrategy,
    SiliconResetStrategy,
)
from src.aiops.transformer_detector import TransformerAnomalyDetector
from src.ml.forecasting.tft_model import PriceTFTModel
from src.shared.observability import (
    post_grafana_annotation,
    push_metrics,
    setup_logging,
)

logger = structlog.get_logger()


class AIOpsOrchestrator:
    def __init__(self, config: dict[str, Any]):
        setup_logging()
        self.config = config
        self.check_interval_seconds = config.get("check_interval_seconds", 60)
        self.prometheus_url = config["prometheus_url"]
        self.api_service_name = config["api_service_name"]

        #  STATEFUL: Remediation tracking and cooldowns
        self.active_remediations = {}
        self.remediation_cooldowns = {
            "retrain": 3600,  # 1 hour cooldown for retraining
            "restart": 300,  # 5 minute cooldown for restarts
            "scale": 120,  # 2 minute cooldown for scaling
            "switch": 600,  # 10 minute cooldown for model swaps
        }
        self.last_remediation_time = {}
        self.error_rate_threshold = config.get("error_rate_threshold", 0.05)
        self.latency_threshold = config.get("latency_threshold", 0.5)
        self.t2t_latency_threshold_ns = config.get(
            "t2t_latency_threshold_ns", 500000
        )  # 500µs
        self.isolation_forest_contamination = config.get(
            "isolation_forest_contamination", 0.1
        )

        # ML Detection Flags
        self.anomaly_detection_enabled = config.get("anomaly_detection_enabled", True)
        self.data_drift_detection_enabled = config.get(
            "data_drift_detection_enabled", True
        )
        self.predictive_scaling_enabled = config.get("predictive_scaling_enabled", True)

        # Initialize Clients
        self.prometheus_client = PrometheusClient(url=self.prometheus_url)

        # Initialize Detectors
        self.isolation_forest_detector = IsolationForestDetector(
            contamination=self.isolation_forest_contamination
        )

        # Forecasting Model for Predictive AIOps
        self.forecaster = (
            PriceTFTModel(config=config.get("tft_config"))
            if self.predictive_scaling_enabled
            else None
        )

        # Autoencoder initialization with logic for disabling
        ae_input_dim = config.get("autoencoder_input_dim")
        if ae_input_dim:  # Changed from 'is not None' to allow 0 to disable
            self.autoencoder_detector = AutoencoderDetector(
                input_dim=ae_input_dim,
                latent_dim=config.get("autoencoder_latent_dim", 2),
                epochs=config.get("autoencoder_epochs", 10),
                threshold_multiplier=config.get(
                    "autoencoder_threshold_multiplier", 2.0
                ),
                verbose=False,
            )
            # Transformer-based Detector
            self.transformer_detector = TransformerAnomalyDetector(
                input_dim=ae_input_dim,
                threshold=config.get("transformer_threshold", 0.05),
            )
        else:
            self.autoencoder_detector = None
            self.transformer_detector = None

        self.data_drift_detector = DataDriftDetector(
            psi_threshold=config.get("data_drift_psi_threshold", 0.1),
            ks_threshold=config.get("data_drift_ks_threshold", 0.05),
        )

        self.docker_remediator = DockerRemediator()
        self.ml_pipeline_trigger = MLPipelineTrigger(
            config=config.get("ml_pipeline_config", {})
        )
        self.redis_remediator = RedisRemediator(
            host=config.get("redis_host", "redis"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
        )
        self.redis_cache_pattern = config.get("redis_cache_pattern", "*")

        #  Strategy Pattern Registry
        self.remediation_registry = RemediationRegistry()
        self._register_default_strategies()

        logger.info("aiops_orchestrator_init", status="success", config=self.config)

    def notify(self, message: str, tags: list[str]):
        """Wraps the async Grafana annotation for sync strategy execution."""
        # This calls the module-level function which the tests patch
        post_grafana_annotation(message, tags)

    def _register_default_strategies(self):
        """Maps anomaly keys to specific remediation strategies."""
        restart = RestartServiceStrategy()
        RetrainModelStrategy()
        purge = PurgeCacheStrategy()
        scale = AutonomousScalerStrategy()
        switch = ModelSwitchStrategy()
        #  SILICON: Performance-specific remediation
        silicon_reset = SiliconResetStrategy()

        self.remediation_registry.register("high_error_rate", restart)
        self.remediation_registry.register("high_latency", switch)
        self.remediation_registry.register("jitter_anomaly", silicon_reset)
        self.remediation_registry.register("data_drift", switch)
        self.remediation_registry.register("univariate_anomaly", purge)
        self.remediation_registry.register("multivariate_anomaly", purge)
        self.remediation_registry.register("transformer_anomaly", purge)
        # Predictive and load-based strategies
        self.remediation_registry.register("predicted_load_spike", scale)
        self.remediation_registry.register("cpu_high", scale)
        self.remediation_registry.register("high_load", scale)

    async def _detect_anomalies(self) -> dict[str, Any]:
        """Scans Prometheus and local detectors for system anomalies."""
        anomalies = {}

        # 1. Threshold-based Error Rate Detection
        error_rate = await self.prometheus_client.get_5xx_error_rate(
            service=self.api_service_name
        )
        if error_rate > self.error_rate_threshold:
            anomalies["high_error_rate"] = {
                "metric": error_rate,
                "threshold": self.error_rate_threshold,
            }

        # 2. Threshold-based Latency Detection
        p95_latency = await self.prometheus_client.get_p95_latency(
            service=self.api_service_name
        )
        if p95_latency > self.latency_threshold:
            anomalies["high_latency"] = {
                "metric": p95_latency,
                "threshold": self.latency_threshold,
                "fallback_model": "black_scholes",  # Safe fallback
                "priority": "high",
            }

        #  HIGH-RES: Tick-to-Trade Jitter Detection
        t2t_latency = await self.prometheus_client.get_custom_metric(
            "bsopt_t2t_latency_ns"
        )
        if t2t_latency > self.t2t_latency_threshold_ns:
            anomalies["jitter_anomaly"] = {
                "metric": t2t_latency,
                "threshold": self.t2t_latency_threshold_ns,
                "priority": "critical",
            }
            logger.error("silicon_jitter_detected", t2t_ns=t2t_latency)

        # 3. Predictive Load Detection via TFT
        if self.predictive_scaling_enabled and self.forecaster:
            # OPTIMIZED: Proactive anomaly detection via forecasting on real range data
            recent_df = await self.prometheus_client.get_metric_range(
                self.api_service_name, "container_cpu_usage_seconds_total"
            )
            if not recent_df.empty and len(recent_df) >= self.forecaster.config.get(
                "max_encoder_length", 24
            ):
                # Predict next 5 minutes
                forecast = self.forecaster.predict(recent_df)
                if forecast is not None and any(
                    forecast > self.latency_threshold * 1.5
                ):  # Predicted spike
                    anomalies["predicted_load_spike"] = {
                        "forecast": forecast.tolist(),
                        "threshold": self.latency_threshold * 1.5,
                        "fallback_model": "xgboost",  # Proactive lighter model
                    }
                    logger.warning(
                        "predictive_anomaly_detected", forecast_max=forecast.max()
                    )

        # 4. ML-Driven Anomaly Detection
        if self.anomaly_detection_enabled:
            # Univariate
            data = await self.prometheus_client.get_historical_metric_data(
                self.api_service_name
            )
            if data is not None and len(data) > 0:
                preds = self.isolation_forest_detector.fit_predict(data)
                if -1 in preds:
                    anomalies["univariate_anomaly"] = {"status": "anomaly_detected"}

            # Multivariate
            if self.autoencoder_detector:
                data_multi = (
                    await self.prometheus_client.get_historical_metric_data_multi(
                        self.api_service_name
                    )
                )
                if data_multi is not None and len(data_multi) > 0:
                    preds_multi = self.autoencoder_detector.fit_predict(data_multi)
                    if -1 in preds_multi:
                        anomalies["multivariate_anomaly"] = {
                            "status": "anomaly_detected"
                        }

                    # Transformer Check
                    if self.transformer_detector:
                        result = self.transformer_detector.detect(data_multi)
                        if result["is_anomaly"]:
                            anomalies["transformer_anomaly"] = {
                                "score": result["score"]
                            }
                            logger.warning(
                                "transformer_anomaly_detected", score=result["score"]
                            )

        # 5. Data Drift Detection
        if self.data_drift_detection_enabled:
            data_multi = await self.prometheus_client.get_historical_metric_data_multi(
                self.api_service_name
            )
            if data_multi is not None and len(data_multi) > 0:
                # Mocking reference vs current for simplicity in this logic
                drift_detected, info = self.data_drift_detector.detect_drift(
                    data_multi, data_multi
                )
                if drift_detected:
                    anomalies["data_drift"] = {
                        "drift_score": info.get("mmd_distance", 0),
                        "fallback_model": "black_scholes",
                    }
            else:
                logger.info("data_drift_check_skipped", reason="no_data")

        return anomalies

    async def _remediate_anomalies(self, anomalies: dict[str, Any]):
        """Dynamic remediation dispatch with stateful cooldowns."""
        current_time = time.time()
        for anomaly_type, data in anomalies.items():
            strategies = self.remediation_registry.get_strategy(anomaly_type)
            for strategy in strategies:
                strategy_name = strategy.__class__.__name__

                # Check cooldowns
                last_time = self.last_remediation_time.get(strategy_name, 0)
                cooldown = self.config.get(
                    f"{anomaly_type}_cooldown", 300
                )  # Default 5m

                if current_time - last_time < cooldown:
                    logger.debug(
                        "remediation_skipped_cooldown",
                        strategy=strategy_name,
                        remaining=cooldown - (current_time - last_time),
                    )
                    continue

                try:
                    logger.info(
                        "executing_remediation",
                        strategy=strategy_name,
                        anomaly=anomaly_type,
                    )
                    await strategy.execute(self, data)
                    self.last_remediation_time[strategy_name] = current_time
                except Exception as e:
                    logger.error(
                        "remediation_execution_failed",
                        strategy=strategy_name,
                        error=str(e),
                    )

    async def run(self, iterations: int = -1):
        iteration_count = 0
        while iterations == -1 or iteration_count < iterations:
            logger.info("aiops_orchestrator_loop", iteration=iteration_count)
            try:
                anomalies = await self._detect_anomalies()
                if anomalies:
                    logger.warning("anomalies_found", detected_anomalies=anomalies)
                    await self._remediate_anomalies(anomalies)
                else:
                    logger.info("no_anomalies_found")
            except Exception as e:
                logger.error("aiops_orchestrator_loop_error", error=str(e))

            push_metrics(job_name="aiops_orchestrator")

            iteration_count += 1
            if iterations == -1 or iteration_count < iterations:
                await asyncio.sleep(self.check_interval_seconds)
