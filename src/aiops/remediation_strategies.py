from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class RemediationStrategy(ABC):
    """Abstract base class for all AIOps remediation strategies."""

    @abstractmethod
    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        pass


class RestartServiceStrategy(RemediationStrategy):
    """Strategy to restart a specific Docker service."""

    supported_types = ["high_latency", "error_spike"]

    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        service_name = anomaly_data.get("service")
        logger.info("remediation_restart_service", service=service_name)
        # Actual implementation would call orchestrator.docker_remediator.restart_service(service_name)


class RetrainModelStrategy(RemediationStrategy):
    """Strategy to trigger ML pipeline retraining."""

    supported_types = ["data_drift", "performance_degradation"]

    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        model_name = anomaly_data.get("model")
        logger.info("remediation_retrain_model", model=model_name)
        # Actual implementation would call orchestrator.ml_trigger.trigger_retraining(model_name)


class PurgeCacheStrategy(RemediationStrategy):
    """Strategy to purge Redis cache."""

    supported_types = ["stale_data", "cache_inconsistency"]

    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        pattern = anomaly_data.get("pattern", "*")
        logger.info("remediation_purge_cache", pattern=pattern)
        # Actual implementation would call orchestrator.redis_remediator.purge_cache(pattern)


class RemediationRegistry:
    """Registry to map anomaly types to remediation strategies."""

    def __init__(self):
        self._strategies: dict[str, RemediationStrategy] = {}

    def register(self, anomaly_type: str, strategy: RemediationStrategy):
        self._strategies[anomaly_type] = strategy

    def get_strategy(self, anomaly_type: str) -> list[RemediationStrategy]:
        """Returns a list of strategies for a given anomaly type."""
        strategy = self._strategies.get(anomaly_type)
        return [strategy] if strategy else []
