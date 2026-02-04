import ray
import asyncio
import structlog
import random
import os
from src.database import get_async_db_context
from sqlalchemy import text

logger = structlog.get_logger(__name__)

class ChaosMonkey:
    """
    SOTA: Proactive failure injection to verify AIOps remediation strategies.
    Only active if BSOPT_CHAOS_MODE=1.
    """
    def __init__(self):
        self.enabled = os.getenv("BSOPT_CHAOS_MODE") == "1"
        if self.enabled:
            logger.warning("chaos_monkey_enabled_prepare_for_disaster")

    def kill_actor(self, actor_name: str):
        """🚀 SINGULARITY: Terminate a random Ray actor matching the name."""
        if not self.enabled: return
        
        try:
            # Find actors by name in the Ray registry (simplified)
            # In production, this would use ray.state to find specific IDs
            logger.error("chaos_injecting_actor_failure", name=actor_name)
            # ray.get_actor(actor_name).exit() 
            # (Simulating failure for the AIOps loop to detect)
            os.environ[f"SIMULATE_FAILURE_{actor_name}"] = "1"
        except Exception as e:
            logger.error("chaos_injection_failed", error=str(e))

    async def delay_db(self, seconds: float = 2.0):
        """🚀 SOTA: Inject latency into a database connection."""
        if not self.enabled: return
        
        logger.error("chaos_injecting_db_latency", seconds=seconds)
        async with get_async_db_context() as session:
            # SOTA: Using pg_sleep to simulate heavy load or network congestion
            await session.execute(text(f"SELECT pg_sleep({seconds})"))

    def partition_network(self, service_url: str):
        """🚀 SOTA: Block traffic to a service URL (simulated)."""
        if not self.enabled: return
        
        logger.error("chaos_injecting_network_partition", url=service_url)
        # In production, this could update an iptables rule or the XDP filter
        os.environ[f"PARTITION_{service_url}"] = "1"

# Global Chaos Engine
monkey = ChaosMonkey()
