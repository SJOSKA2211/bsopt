import asyncio
from typing import Any

import structlog

from src.shared.config import settings

logger = structlog.get_logger(__name__)

class AutonomousGuardian:
    """
    High-level supervisor for the BS-OPT autonomous manifold.
    Enforces risk-based circuit breakers and coordinates cross-component remediation.
    """

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self.is_active = True
        self.is_safe_mode = False
        self.risk_threshold = settings.MAX_NET_DELTA * 0.95
        self.drift_critical_threshold = 0.5  # PSI critical level
        self.paused_features = []

    async def monitor_integrity(self):
        """Continuous oversight of system integrity and risk levels."""
        logger.info("autonomous_guardian_activated")
        
        while self.is_active:
            try:
                # 1. Check for Cascading Service Failures
                await self._check_cascading_failures()

                # 2. Check Drift Levels in History
                for event in self.orchestrator.history:
                    if event.get("anomaly") == "distribution_drift" and event.get("score", 0) > self.drift_critical_threshold:
                        logger.critical("guardian_critical_drift_detected", score=event.get("score"))
                        await self.halt_non_critical_operations("critical_drift")
                
                # 3. Check Risk Exposure (Live check would go here)
                pass

            except Exception as e:
                logger.error("guardian_oversight_error", error=str(e))
            
            await asyncio.sleep(60)

    async def _check_cascading_failures(self):
        """Detects if multiple core services are degraded simultaneously."""
        if not self.orchestrator.health_reporter:
            return

        report = await self.orchestrator.health_reporter.get_health_report()
        degraded_count = 0
        failed_services = []

        if not report.rabbitmq.connected:
            degraded_count += 1
            failed_services.append("rabbitmq")
        if not report.redis.connected:
            degraded_count += 1
            failed_services.append("redis")
        if not report.postgres.connected:
            degraded_count += 1
            failed_services.append("postgres")

        if degraded_count >= 2:
            logger.critical("guardian_cascading_failure_detected", services=failed_services)
            if not self.is_safe_mode:
                await self.activate_safe_mode(failed_services)
        elif degraded_count == 0 and self.is_safe_mode:
            await self.deactivate_safe_mode()

    async def activate_safe_mode(self, reasons: list[str]):
        """Engages protective circuit breakers."""
        logger.warning("guardian_activating_safe_mode", reasons=reasons)
        self.is_safe_mode = True
        await self.halt_non_critical_operations("cascading_failure")

    async def deactivate_safe_mode(self):
        """Disengages protective circuit breakers."""
        logger.info("guardian_deactivating_safe_mode_system_stable")
        self.is_safe_mode = False
        self.paused_features = []
        from src.shared.utils.cache import get_redis
        redis = get_redis()
        if redis:
            await redis.delete("bsopt:trading:paused")

    async def halt_non_critical_operations(self, reason: str):
        """Triggers a protective shutdown of high-risk autonomous features."""
        if "trading" not in self.paused_features:
            logger.warning("guardian_initiating_protective_cutoff", reason=reason)
            from src.shared.utils.cache import get_redis
            redis = get_redis()
            if redis:
                await redis.set("bsopt:trading:paused", "true", ex=3600)
                self.paused_features.append("trading")
                logger.info("trading_autonomy_paused_via_guardian")

    def stop(self):
        self.is_active = False
        logger.info("autonomous_guardian_deactivated")
