import structlog
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

logger = structlog.get_logger()

class RemediationStrategy(ABC):
    """Abstract base class for all AIOps remediation strategies."""
    @abstractmethod
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
        pass

class RestartServiceStrategy(RemediationStrategy):
    """Strategy to restart a specific Docker service."""
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
        message = f"Remediation: Restarting '{orchestrator.api_service_name}' due to anomaly."
        logger.info("remediation_action", action="restart_service", service=orchestrator.api_service_name, message=message)
        orchestrator.docker_remediator.restart_service(orchestrator.api_service_name)
        orchestrator.notify(message, ["remediation", "api_spike", orchestrator.api_service_name])

class RetrainModelStrategy(RemediationStrategy):
    """Strategy to trigger ML pipeline retraining."""
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
        message = "Remediation: Triggering ML pipeline retraining due to data drift."
        logger.info("remediation_action", action="trigger_ml_retraining", message=message)
        orchestrator.ml_pipeline_trigger.trigger_retraining()
        orchestrator.notify(message, ["remediation", "data_drift"])

class PurgeCacheStrategy(RemediationStrategy):
    """Strategy to purge Redis cache."""
    def execute(self, orchestrator: Any, anomaly_data: Dict[str, Any]):
        message = f"Remediation: Purging Redis cache due to anomaly."
        logger.info("remediation_action", action="purge_redis_cache", pattern=orchestrator.redis_cache_pattern, message=message)
        orchestrator.redis_remediator.purge_cache(orchestrator.redis_cache_pattern)
        orchestrator.notify(message, ["remediation", "anomaly", "redis_cache"])

class RemediationRegistry:
    """Registry to map anomaly types to remediation strategies."""
    def __init__(self):
        self._strategies: Dict[str, RemediationStrategy] = {}

    def register(self, anomaly_type: str, strategy: RemediationStrategy):
        self._strategies[anomaly_type] = strategy

    def get_strategy(self, anomaly_type: str) -> List[RemediationStrategy]:
        """Returns a list of strategies for a given anomaly type."""
        strategy = self._strategies.get(anomaly_type)
        return [strategy] if strategy else []
