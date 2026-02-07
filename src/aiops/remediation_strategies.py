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
        service_name = orchestrator.api_service_name
        logger.warning("remediation_restart_service_trigger", service=service_name)
        orchestrator.docker_remediator.restart_service(service_name)
        orchestrator.notify(f"AIOps: Restarted service {service_name} due to anomaly.", ["aiops", "remediation", "restart"])

class RetrainModelStrategy(RemediationStrategy):
    """Strategy to trigger ML pipeline retraining."""
    supported_types = ["data_drift", "performance_degradation"]
    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        logger.warning("remediation_retrain_model_trigger")
        orchestrator.ml_pipeline_trigger.trigger_retraining()
        orchestrator.notify("AIOps: Triggered ML pipeline retraining due to data drift.", ["aiops", "remediation", "retrain"])

class PurgeCacheStrategy(RemediationStrategy):
    """Strategy to purge Redis cache."""
    supported_types = ["stale_data", "cache_inconsistency"]
    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        pattern = orchestrator.redis_cache_pattern
        logger.warning("remediation_purge_cache_trigger", pattern=pattern)
        orchestrator.redis_remediator.purge_cache(pattern)
        orchestrator.notify(f"AIOps: Purged Redis cache pattern {pattern} due to anomaly.", ["aiops", "remediation", "purge"])

class ArgoCDRollbackStrategy(RemediationStrategy):
    """Strategy to roll back a deployment via ArgoCD."""
    supported_types = ["deployment_regression", "sync_failure"]
    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        from src.aiops.argocd_remediator import ArgoCDRollbackRemediator
        remediator = ArgoCDRollbackRemediator()
        logger.warning("remediation_argocd_rollback_trigger", app=anomaly_data.get("service"))
        remediator.remediate(anomaly_data)
        orchestrator.notify(f"AIOps: Triggered ArgoCD rollback for {anomaly_data.get('service')} due to regression.", ["aiops", "remediation", "rollback"])

class AutonomousScalerStrategy(RemediationStrategy):
    """Strategy to autonomously scale a service based on load or predicted spikes."""
    supported_types = ["high_load", "predicted_load_spike", "cpu_high"]
    def execute(self, orchestrator: Any, anomaly_data: dict[str, Any]):
        service_name = orchestrator.api_service_name
        current_replicas = anomaly_data.get("metrics", {}).get("replicas", 1)
        target_replicas = current_replicas + 1
        
        logger.warning("remediation_scaling_trigger", service=service_name, target=target_replicas)
        success = orchestrator.docker_remediator.scale_service(service_name, target_replicas)
        if success:
            orchestrator.notify(f"AIOps: Autonomously scaled {service_name} to {target_replicas} replicas.", ["aiops", "remediation", "scale"])


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
