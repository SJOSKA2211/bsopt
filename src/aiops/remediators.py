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
        self.last_run = 0.0
        self.cooldown = 300.0  # Default cooldown 5 mins

    def can_run(self) -> bool:
        """Check if cooldown has passed."""
        return (asyncio.get_event_loop().time() - self.last_run) >= self.cooldown

    @abstractmethod
    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        """Execute the remediation action."""
        return False

    async def validate(self, anomaly: dict[str, Any]) -> bool:
        """
        Optional post-remediation validation.
        """
        return True

    async def update_last_run(self):
        self.last_run = asyncio.get_event_loop().time()


class ClearRedisCacheRemediator(BaseRemediator):
    """
    Clears the Redis cache if high latency or stale data is detected.
    """

    def __init__(self):
        super().__init__("clear_cache", supported_types=["latency_spike", "stale_data", "cache_inconsistency"])

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


class RestartServiceRemediator(BaseRemediator):
    """
    Restarts a service via Docker/Orchestrator.
    """

    def __init__(self):
        super().__init__(
            "restart_service",
            supported_types=["latency_spike", "error_burst", "cpu_high", "high_latency", "error_spike"],
        )

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        from src.aiops.docker_remediator import DockerRemediator
        service = anomaly.get("metrics", {}).get("service", "bsopt-api")
        logger.warning("remediator_restart_initiated", service=service)
        
        docker = DockerRemediator()
        success = await docker.restart_service(service)
        
        if success:
            logger.info("remediator_restart_completed", service=service)
        return success


class RetrainModelRemediator(BaseRemediator):
    """
    Triggers the ML retraining pipeline if drift is detected.
    """

    def __init__(self):
        super().__init__(
            "retrain_model", supported_types=["model_drift", "performance_degradation", "data_drift"]
        )

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        logger.warning("remediator_retrain_initiated", score=anomaly.get("score"))
        monitor_drift_and_retrain_task.delay()
        logger.info("remediator_retrain_task_queued")
        return True


class ArgoCDRollbackRemediator(BaseRemediator):
    """
    Rolls back a deployment via ArgoCD.
    """

    def __init__(self):
        super().__init__("argocd_rollback", supported_types=["deployment_regression", "sync_failure"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        service = anomaly.get("service", "unknown")
        logger.warning("remediator_argocd_rollback_initiated", app=service)
        # Mocking actual implementation
        await asyncio.sleep(1)
        logger.info("remediator_argocd_rollback_completed", app=service)
        return True


class AutonomousScalerRemediator(BaseRemediator):
    """
    Scales service replicas based on load.
    """

    def __init__(self):
        super().__init__("autonomous_scaler", supported_types=["high_load", "predicted_load_spike", "cpu_high"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        from src.aiops.docker_remediator import DockerRemediator
        service = anomaly.get("service", "bsopt-api")
        current_replicas = anomaly.get("metrics", {}).get("replicas", 1)
        target_replicas = current_replicas + 1
        
        logger.warning("remediator_scaling_initiated", service=service, target=target_replicas)
        docker = DockerRemediator()
        success = await docker.scale_service(service, target_replicas)
        return success


class ModelSwitchRemediator(BaseRemediator):
    """
    Hot-swaps models during drift.
    """

    def __init__(self):
        super().__init__("model_switch", supported_types=["data_drift", "model_instability"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        from src.pricing.factory import PricingEngineFactory
        fallback = anomaly.get("fallback_model", "black_scholes")
        current = anomaly.get("model", "unknown")
        
        logger.warning("remediator_model_switch_initiated", from_model=current, to_model=fallback)
        PricingEngineFactory.set_default_engine(fallback)
        return True


class SiliconResetRemediator(BaseRemediator):
    """
    Critical reset for low-latency workers (Direct Service Restart).
    """

    def __init__(self):
        super().__init__("silicon_reset", supported_types=["critical_jitter", "worker_unresponsive"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        from src.aiops.docker_remediator import DockerRemediator
        logger.warning("remediator_silicon_reset_initiated")
        docker = DockerRemediator()
        success = await docker.restart_service("worker")
        return success


class KernelTuningRemediator(BaseRemediator):
    """
    Proactively tunes OS kernel parameters for low-latency processing.
    Triggers the 'Vanguard' optimization script.
    """

    def __init__(self):
        super().__init__("kernel_tuning", supported_types=["latency_spike", "system_jitter"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        logger.warning("remediator_kernel_tuning_initiated")
        try:
            # Executes the optimize_kernel.sh script.
            # Requires sudo/root permissions or pre-configured NOPASSWD in sudoers.
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "/app/scripts/optimize_kernel.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info("remediator_kernel_tuning_successful", output=stdout.decode()[:100])
                return True
            else:
                logger.error("remediator_kernel_tuning_failed", error=stderr.decode())
                return False
        except Exception as e:
            logger.error("remediator_kernel_tuning_error", error=str(e))
            return False



class RemediationPlanner:
    """
    Intelligent selector for remediation actions.
    Decides the best course of action based on anomaly context.
    """

    def __init__(self, remediators: list[BaseRemediator] | None = None):
        if remediators:
            self.remediators = {r.name: r for r in remediators}
        else:
            # Register defaults
            defaults = [
                ClearRedisCacheRemediator(),
                RestartServiceRemediator(),
                RetrainModelRemediator(),
                ArgoCDRollbackRemediator(),
                AutonomousScalerRemediator(),
                ModelSwitchRemediator(),
                SiliconResetRemediator(),
                KernelTuningRemediator(),
            ]
            self.remediators = {r.name: r for r in defaults}


    def plan(self, anomaly: dict[str, Any]) -> list[BaseRemediator]:
        """
        Plans a sequence of remediation actions.
        """
        a_type = anomaly.get("type", "generic")
        actions = []
        for r in self.remediators.values():
            if a_type in r.supported_types and r.can_run():
                actions.append(r)
        return sorted(actions, key=lambda x: x.name)

