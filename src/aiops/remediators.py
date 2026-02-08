import asyncio
from abc import ABC, abstractmethod
from typing import Any

import structlog

from src.tasks.ml_tasks import monitor_drift_and_retrain_task

logger = structlog.get_logger(__name__)

class BaseRemediator(ABC):
    """
    Abstract base class for all automated remediation actions.
    Implements common logic like validation and backoff.
    """
    def __init__(self, name: str, supported_types: list[str] | None = None):
        self.name = name
        self.supported_types = supported_types or ["generic"]
        self.last_run = 0
        self.cooldown = 60 # Default cooldown in seconds

    @abstractmethod
    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        """Execute the remediation action."""
        pass

    async def validate(self, anomaly: dict[str, Any]) -> bool:
        """
        Optional post-remediation validation.
        Should return True if the system is 'healed'.
        """
        return True

class RestartServiceRemediator(BaseRemediator):
    """
    Simulates restarting a service. 
    In a real production environment, this would call Docker/K8s APIs.
    """
    def __init__(self):
        super().__init__("restart_service", supported_types=["latency_spike", "error_burst", "cpu_high"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        service = anomaly.get("metrics", {}).get("service", "unknown")
        logger.warning("remediator_restart_initiated", service=service)
        
        # Simulate restart delay
        await asyncio.sleep(2)
        
        logger.info("remediator_restart_completed", service=service)
        return True

class RetrainModelRemediator(BaseRemediator):
    """
    Triggers the ML retraining pipeline if drift is detected.
    """
    def __init__(self):
        super().__init__("retrain_model", supported_types=["model_drift", "performance_degradation"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        logger.warning("remediator_retrain_initiated", score=anomaly.get("score"))
        
        # Trigger the Celery task for retraining
        monitor_drift_and_retrain_task.delay()
        
        logger.info("remediator_retrain_task_queued")
        return True

class RemediationPlanner:
    """
    Intelligent selector for remediation actions.
    Decides the best course of action based on anomaly context.
    """
    def __init__(self, remediators: list[BaseRemediator]):
        self.remediators = {r.name: r for r in remediators}

    def plan(self, anomaly: dict[str, Any]) -> list[BaseRemediator]:
        """
        Plans a sequence of remediation actions.
        """
        a_type = anomaly.get("type", "generic")
        
        actions = []
        
        # Simple heuristic-based planning
        for r in self.remediators.values():
            if a_type in r.supported_types:
                actions.append(r)
                
        # Prioritize actions based on score (placeholder for more complex logic)
        return sorted(actions, key=lambda x: x.name)
