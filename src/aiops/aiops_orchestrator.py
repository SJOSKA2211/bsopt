import time
import os
import structlog
from typing import Dict, Any
from src.aiops.prometheus_adapter import PrometheusClient
from src.aiops.isolation_forest_detector import IsolationForestDetector
from src.aiops.autoencoder_detector import AutoencoderDetector
from src.aiops.data_drift_detector import DataDriftDetector
from src.aiops.docker_remediator import DockerRemediator
from src.aiops.ml_pipeline_trigger import MLPipelineTrigger
from src.aiops.redis_remediator import RedisRemediator
from src.shared.observability import setup_logging, push_metrics, post_grafana_annotation

logger = structlog.get_logger()

class AIOpsOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        setup_logging()
        self.config = config
        self.check_interval_seconds = config.get("check_interval_seconds", 60)
        self.prometheus_url = config["prometheus_url"]
        self.api_service_name = config["api_service_name"]

        # Anomaly Detection Thresholds
        self.error_rate_threshold = config.get("error_rate_threshold", 0.05)
        self.latency_threshold = config.get("latency_threshold", 0.5) # seconds
        self.isolation_forest_contamination = config.get("isolation_forest_contamination", 0.1)
        self.autoencoder_input_dim = config.get("autoencoder_input_dim")
        self.autoencoder_latent_dim = config.get("autoencoder_latent_dim")
        self.autoencoder_epochs = config.get("autoencoder_epochs", 10)
        self.autoencoder_threshold_multiplier = config.get("autoencoder_threshold_multiplier", 2.0)
        self.data_drift_psi_threshold = config.get("data_drift_psi_threshold", 0.1)
        self.data_drift_ks_threshold = config.get("data_drift_ks_threshold", 0.05)

        # Feature Flags for Anomaly Detection
        self.anomaly_detection_enabled = config.get("anomaly_detection_enabled", False)
        self.data_drift_detection_enabled = config.get("data_drift_detection_enabled", False)
        
        # Remediation Actions
        self.ml_pipeline_config = config.get("ml_pipeline_config", {})
        self.redis_cache_pattern = config.get("redis_cache_pattern", "*")

        # Initialize Clients and Detectors
        self.prometheus_client = PrometheusClient(url=self.prometheus_url)
        self.isolation_forest_detector = IsolationForestDetector(contamination=self.isolation_forest_contamination)
        
        # Autoencoder needs input_dim to be initialized
        if self.autoencoder_input_dim:
            self.autoencoder_detector = AutoencoderDetector(
                input_dim=self.autoencoder_input_dim,
                latent_dim=self.autoencoder_latent_dim,
                epochs=self.autoencoder_epochs,
                threshold_multiplier=self.autoencoder_threshold_multiplier,
                verbose=False
            )
        else:
            self.autoencoder_detector = None # Will not be used if input_dim is not provided

        self.data_drift_detector = DataDriftDetector(
            psi_threshold=self.data_drift_psi_threshold,
            ks_threshold=self.data_drift_ks_threshold
        )
        self.docker_remediator = DockerRemediator()
        self.ml_pipeline_trigger = MLPipelineTrigger(config=self.ml_pipeline_config)
        self.redis_remediator = RedisRemediator()

        logger.info("aiops_orchestrator_init", status="success", config=self.config)

    def _detect_anomalies(self) -> Dict[str, Any]:
        anomalies = {}

        # Prometheus Metrics Checks
        error_rate = self.prometheus_client.get_5xx_error_rate(service=self.api_service_name)
        latency = self.prometheus_client.get_p95_latency(service=self.api_service_name)
        
        if error_rate >= self.error_rate_threshold:
            anomalies["high_error_rate"] = True
            logger.warning("anomaly_detected", type="high_error_rate", service=self.api_service_name, value=error_rate, threshold=self.error_rate_threshold)
        if latency >= self.latency_threshold:
            anomalies["high_latency"] = True
            logger.warning("anomaly_detected", type="high_latency", service=self.api_service_name, value=latency, threshold=self.latency_threshold)
        
        
        # ML-driven Anomaly Detection (using dummy data for now)
        if self.anomaly_detection_enabled:
            # Univariate (e.g., latency, error rate over time)
            # For actual implementation, would fetch historical data
            dummy_univariate_data = self.prometheus_client.get_historical_metric_data( # Use self.prometheus_client
                query=f'http_requests_5xx_rate{{service="{self.api_service_name}"}}[5m]',
                duration_seconds=300
            )
            # Correctly check for empty numpy array
            if dummy_univariate_data.size > 0:
                anomalies_univariate = self.isolation_forest_detector.fit_predict(dummy_univariate_data)
                if -1 in anomalies_univariate:
                    anomalies["univariate_anomaly"] = True
                    logger.warning("anomaly_detected", type="univariate_anomaly", detector="IsolationForest")

            # Multivariate (e.g., multiple metrics together)
            if self.autoencoder_detector and self.autoencoder_input_dim:
                # For actual implementation, fetch multiple historical metrics
                dummy_multivariate_data = self.prometheus_client.get_historical_metric_data_multi( # Use self.prometheus_client
                    queries=[
                        f'http_requests_5xx_rate{{service="{self.api_service_name}"}}[5m]',
                        f'http_request_duration_seconds_bucket{{service="{self.api_service_name}"}}[5m]'
                    ],
                    duration_seconds=300
                )
                if dummy_multivariate_data.size > 0: # Correctly check for empty numpy array
                    anomalies_multivariate = self.autoencoder_detector.fit_predict(dummy_multivariate_data)
                    if -1 in anomalies_multivariate:
                        anomalies["multivariate_anomaly"] = True
                        logger.warning("anomaly_detected", type="multivariate_anomaly", detector="Autoencoder")

        # ML Data Drift Detection (using dummy data for now)
        if self.data_drift_detection_enabled:
            # For actual implementation, would fetch real reference and current data
            dummy_reference_data = self.prometheus_client.get_historical_metric_data_multi( # Use self.prometheus_client
                queries=[f'metric_feature_1{{service="{self.api_service_name}"}}[1h]'],
                duration_seconds=3600 # 1 hour of data
            )
            dummy_current_data = self.prometheus_client.get_historical_metric_data_multi( # Use self.prometheus_client
                queries=[f'metric_feature_1{{service="{self.api_service_name}"}}[5m]'],
                duration_seconds=300 # 5 minutes of data
            )
            
            # Ensure data is not empty before passing to detector
            if dummy_reference_data.size > 0 and dummy_current_data.size > 0: # Correctly check for empty numpy array
                is_drifted, drift_info = self.data_drift_detector.detect_drift(dummy_reference_data, dummy_current_data)
                if is_drifted:
                    anomalies["data_drift"] = True
                    logger.warning("anomaly_detected", type="data_drift", info=drift_info)
            else:
                logger.info("data_drift_check_skipped", reason="insufficient_data")

        return anomalies

    def _remediate_anomalies(self, anomalies: Dict[str, Any]):
        if anomalies.get("high_error_rate") or anomalies.get("high_latency"):
            message = f"Remediation: Restarting '{self.api_service_name}' due to {'high error rate' if anomalies.get('high_error_rate') else 'high latency'}."
            logger.info("remediation_action", action="restart_service", service=self.api_service_name, message=message)
            self.docker_remediator.restart_service(self.api_service_name)
            post_grafana_annotation(message, ["remediation", "api_spike", self.api_service_name])
        
        if anomalies.get("data_drift"):
            message = "Remediation: Triggering ML pipeline retraining due to data drift."
            logger.info("remediation_action", action="trigger_ml_retraining", message=message)
            self.ml_pipeline_trigger.trigger_retraining()
            post_grafana_annotation(message, ["remediation", "data_drift"]) # Add relevant tags for ML drift

        if anomalies.get("univariate_anomaly") or anomalies.get("multivariate_anomaly"):
            message = f"Remediation: Purging Redis cache due to {'univariate' if anomalies.get('univariate_anomaly') else 'multivariate'} anomaly."
            logger.info("remediation_action", action="purge_redis_cache", pattern=self.redis_cache_pattern, message=message)
            self.redis_remediator.purge_cache(self.redis_cache_pattern)
            post_grafana_annotation(message, ["remediation", "anomaly", "redis_cache"]) # Add relevant tags for anomaly

    def run(self, iterations: int = -1):
        iteration_count = 0
        while iterations == -1 or iteration_count < iterations:
            logger.info("aiops_orchestrator_loop", iteration=iteration_count)
            try:
                anomalies = self._detect_anomalies()
                if anomalies:
                    logger.warning("anomalies_found", detected_anomalies=anomalies)
                    self._remediate_anomalies(anomalies)
                else:
                    logger.info("no_anomalies_found")
            except Exception as e:
                logger.error("aiops_orchestrator_loop_error", error=str(e))
            
            push_metrics(job_name="aiops_orchestrator") # Push internal metrics
            
            iteration_count += 1
            if iterations == -1 or iteration_count < iterations:
                time.sleep(self.check_interval_seconds)

if __name__ == "__main__": # pragma: no cover
    # Example usage - replace with actual configuration
    config = {
        "check_interval_seconds": 5,
        "prometheus_url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        "api_service_name": os.getenv("API_SERVICE_NAME", "api"),
        "error_rate_threshold": float(os.getenv("ERROR_RATE_THRESHOLD", "0.05")),
        "latency_threshold": float(os.getenv("LATENCY_THRESHOLD", "0.5")),
        "isolation_forest_contamination": float(os.getenv("ISOLATION_FOREST_CONTAMINATION", "0.1")),
        "autoencoder_input_dim": int(os.getenv("AUTOENCODER_INPUT_DIM", "5")), # Example: 5 metrics for multivariate
        "autoencoder_latent_dim": int(os.getenv("AUTOENCODER_LATENT_DIM", "2")),
        "autoencoder_epochs": int(os.getenv("AUTOENCODER_EPOCHS", "10")),
        "autoencoder_threshold_multiplier": float(os.getenv("AUTOENCODER_THRESHOLD_MULTIPLIER", "2.0")),
        "data_drift_psi_threshold": float(os.getenv("DATA_DRIFT_PSI_THRESHOLD", "0.1")),
        "data_drift_ks_threshold": float(os.getenv("DATA_DRIFT_KS_THRESHOLD", "0.05")),
        "anomaly_detection_enabled": os.getenv("ANOMALY_DETECTION_ENABLED", "True").lower() == "true",
        "data_drift_detection_enabled": os.getenv("DATA_DRIFT_DETECTION_ENABLED", "True").lower() == "true",
        "ml_pipeline_config": {
            "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", "DEMO_KEY"),
            "db_url": os.getenv("DATABASE_URL", "sqlite:///bsopt.db"),
            "ticker": os.getenv("ML_PIPELINE_TICKER", "AAPL"),
            "study_name": os.getenv("ML_PIPELINE_STUDY_NAME", "aiops_retrain"),
            "framework": os.getenv("ML_PIPELINE_FRAMEWORK", "xgboost")
        },
        "redis_cache_pattern": os.getenv("REDIS_CACHE_PATTERN", "aiops_cache:*")
    }

    orchestrator = AIOpsOrchestrator(config)
    orchestrator.run()
