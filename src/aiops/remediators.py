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
<<<<<<< Updated upstream
        self.last_run = 0
        self.cooldown = 60 # Default cooldown in seconds
=======
        self.last_run = 0.0
        self.cooldown = 300.0  # Default cooldown 5 mins

    def can_run(self) -> bool:
        """Check if cooldown has passed."""
        return (asyncio.get_event_loop().time() - self.last_run) >= self.cooldown
>>>>>>> Stashed changes

    @abstractmethod
    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        """Execute the remediation action."""
        pass

<<<<<<< Updated upstream
    async def validate(self, anomaly: dict[str, Any]) -> bool:
        """
        Optional post-remediation validation.
        Should return True if the system is 'healed'.
        """
        return True
=======
    async def update_last_run(self):
        self.last_run = asyncio.get_event_loop().time()


class ClearRedisCacheRemediator(BaseRemediator):
    """
    Clears the Redis cache if high latency or stale data is detected.
    """

    def __init__(self):
        super().__init__("clear_cache", supported_types=["latency_spike", "stale_data"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        import redis.asyncio as redis

        from src.config import settings

        logger.warning("remediator_clear_cache_initiated")
        try:
            client = redis.from_url(settings.REDIS_URL)
            await client.flushdb()
            logger.info("remediator_clear_cache_completed")
            return True
        except Exception as e:
            logger.error("remediator_clear_cache_failed", error=str(e))
            return False
>>>>>>> Stashed changes


class RestartServiceRemediator(BaseRemediator):
    """
    Simulates restarting a service.
    In a real production environment, this would call Docker/K8s APIs.
    """

    def __init__(self):
        super().__init__(
            "restart_service",
            supported_types=["latency_spike", "error_burst", "cpu_high"],
        )

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
        super().__init__(
            "retrain_model", supported_types=["model_drift", "performance_degradation"]
        )

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
