import os

import structlog
from sqlalchemy import text

from src.database import get_async_db_context

logger = structlog.get_logger(__name__)


class ChaosMonkey:
    """
    OPTIMIZED: Proactive failure injection to verify AIOps remediation strategies.
    Only active if BSOPT_CHAOS_MODE=1.
    """

    def __init__(self):
        from src.shared.config import settings

        self.enabled = settings.CHAOS_MODE
        if self.enabled:
            logger.warning("chaos_monkey_enabled_prepare_for_disaster")

    def kill_actor(self, actor_name: str):
        """Terminate a random Ray actor matching the name."""
        if not self.enabled:
            return

        try:
            # FIND: Find actors by name in the Ray registry
            # REAL: Proactive termination to verify AIOps remediation
            logger.error("chaos_injecting_actor_failure", name=actor_name)
            try:
                ray.get_actor(actor_name).exit()
            except Exception:
                # Fallback to process-level signaling if actor handle was lost
                os.environ[f"SIMULATE_FAILURE_{actor_name}"] = "1"
        except Exception as e:
            logger.error("chaos_injection_failed", error=str(e))

    async def delay_db(self, seconds: float = 2.0):
        """Inject latency into a database connection."""
        if not self.enabled:
            return

        logger.error("chaos_injecting_db_latency", seconds=seconds)
        async with get_async_db_context() as session:
            await session.execute(text("SELECT pg_sleep(:seconds)"), {"seconds": seconds})

    def partition_network(self, service_url: str):
        """Block traffic to a service URL (simulated)."""
        if not self.enabled:
            return

        logger.error("chaos_injecting_network_partition", url=service_url)
        # In production, this could update an iptables rule or the XDP filter
        os.environ[f"PARTITION_{service_url}"] = "1"


# Global Chaos Engine
monkey = ChaosMonkey()
