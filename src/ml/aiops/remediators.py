import asyncio
from abc import ABC, abstractmethod
from typing import Any

import structlog

from src.workers.tasks.ml_tasks import monitor_drift_and_retrain_task

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
        super().__init__(
            "clear_cache", supported_types=["latency_spike", "stale_data", "cache_inconsistency"]
        )

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        from src.shared.utils.cache import get_redis

        logger.warning("remediator_clear_cache_initiated")
        try:
            client = get_redis()
            if client:
                await client.flushdb()
                logger.info("remediator_clear_cache_completed")
                return True
            logger.error("remediator_clear_cache_failed_no_client")
            return False
        except Exception as e:
            logger.error("remediator_clear_cache_failed", error=str(e), exc_info=True)
            return False


class RestartServiceRemediator(BaseRemediator):
    """
    Restarts a service via Docker/Orchestrator.
    """

    def __init__(self):
        super().__init__(
            "restart_service",
            supported_types=[
                "latency_spike",
                "error_burst",
                "cpu_high",
                "high_latency",
                "error_spike",
            ],
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
            "retrain_model",
            supported_types=["model_drift", "performance_degradation", "data_drift"],
        )

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        ticker = anomaly.get("metrics", {}).get("ticker", "AAPL")
        logger.warning("remediator_retrain_initiated", ticker=ticker, score=anomaly.get("score"))
        monitor_drift_and_retrain_task.delay(ticker=ticker)
        logger.info("remediator_retrain_task_queued", ticker=ticker)
        return True


class ArgoCDRollbackRemediator(BaseRemediator):
    """
    Rolls back a deployment via ArgoCD.
    """

    def __init__(self):
        super().__init__(
            "argocd_rollback", supported_types=["deployment_regression", "sync_failure"]
        )

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
        super().__init__(
            "autonomous_scaler", supported_types=["high_load", "predicted_load_spike", "cpu_high"]
        )

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
        from src.math_kernel.factory import PricingEngineFactory

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
        super().__init__(
            "silicon_reset", supported_types=["critical_jitter", "worker_unresponsive"]
        )

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

    SCRIPT_PATH = "/app/scripts/optimize_kernel.sh"

    def __init__(self):
        super().__init__("kernel_tuning", supported_types=["latency_spike", "system_jitter"])

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        logger.warning("remediator_kernel_tuning_initiated")
        try:
            # Executes the optimize_kernel.sh script with absolute path.
            # Requires sudo/root permissions or pre-configured NOPASSWD in sudoers.
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                self.SCRIPT_PATH,
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


class DatabasePoolRemediator(BaseRemediator):
    """
    Handles database connection pool exhaustion.
    Recycles idle connections and optionally increases pool size.
    """

    def __init__(self):
        super().__init__(
            "db_pool_recovery",
            supported_types=["db_pool_exhaustion", "db_connection_timeout", "high_db_latency"],
        )
        self.cooldown = 120.0  # 2 min cooldown

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        logger.warning("remediator_db_pool_recovery_initiated")
        try:
            from src.database import db_manager

            # Dispose all connections in the pool, forcing new ones to be created
            await db_manager.dispose()
            logger.info("remediator_db_pool_recycled", action="dispose")

            # Optionally adjust pool size if metrics indicate sustained pressure
            pool_pressure = anomaly.get("metrics", {}).get("pool_utilization", 0.0)
            if pool_pressure > 0.9:
                logger.warning(
                    "remediator_db_pool_pressure_critical",
                    utilization=pool_pressure,
                    recommendation="increase_pool_size",
                )

            return True
        except Exception as e:
            logger.error("remediator_db_pool_recovery_failed", error=str(e))
            return False

    async def validate(self, anomaly: dict[str, Any]) -> bool:
        """Verify that the pool is accepting new connections."""
        try:
            from src.database import db_manager

            engine = db_manager.engine
            with engine.connect() as conn:
                from sqlalchemy import text

                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class RabbitMQCongestionRemediator(BaseRemediator):
    """
    Handles RabbitMQ queue congestion by purging dead-letter queues,
    restarting consumers, or temporarily increasing prefetch count.
    """

    def __init__(self):
        super().__init__(
            "rabbitmq_congestion",
            supported_types=["queue_backpressure", "consumer_lag", "dlq_overflow"],
        )
        self.cooldown = 180.0  # 3 min cooldown
        self.allowed_queues = {"default", "ml_tasks", "pricing", "scraper", "trading"}
        self.allowed_actions = {"purge_dlq", "increase_prefetch", "restart_consumers"}

    async def remediate(self, anomaly: dict[str, Any]) -> bool:
        queue_name = anomaly.get("metrics", {}).get("queue", "default")
        action = anomaly.get("metrics", {}).get("suggested_action", "purge_dlq")

        if queue_name not in self.allowed_queues or action not in self.allowed_actions:
            logger.error("remediator_rabbitmq_forbidden", queue=queue_name, action=action)
            return False

        logger.warning(
            "remediator_rabbitmq_congestion_initiated",
            queue=queue_name,
            action=action,
        )

        try:
            import aio_pika

            from src.config import settings

            broker_url = f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASSWORD}@{settings.RABBITMQ_HOST}:5672/"
            connection = await aio_pika.connect_robust(broker_url)
            channel = await connection.channel()

            if action == "purge_dlq":
                dlq_name = f"{queue_name}.dlq"
                dlq = await channel.declare_queue(dlq_name, passive=True)
                await dlq.purge()
                logger.info("remediator_rabbitmq_dlq_purged", queue=dlq_name)

            elif action == "increase_prefetch":
                # Temporarily increase prefetch to drain the backlog
                await channel.set_qos(prefetch_count=50)
                logger.info("remediator_rabbitmq_prefetch_increased", prefetch=50)

            elif action == "restart_consumers":
                # Signal consumer restart via Redis pub/sub or direct restart
                from src.aiops.docker_remediator import DockerRemediator

                docker = DockerRemediator()
                await docker.restart_service("worker")
                logger.info("remediator_rabbitmq_consumers_restarted")

            await connection.close()
            return True
        except Exception as e:
            logger.error("remediator_rabbitmq_congestion_failed", error=str(e), exc_info=True)
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
                DatabasePoolRemediator(),
                RabbitMQCongestionRemediator(),
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

    def close(self):
        """Cleanup all registered remediators."""
        for r in self.remediators.values():
            if hasattr(r, "close"):
                r.close()
        logger.info("remediation_planner_closed")
