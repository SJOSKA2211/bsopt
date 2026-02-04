import asyncio
import os
import structlog
from typing import Dict, Any
from src.shared.observability import get_obs_client
from src.config import settings

logger = structlog.get_logger()

class LatencyRemediator:
    """
    Proactively tunes the OS and Platform based on real-time latency metrics.
    Connects the 'Omniscient' observability to the 'Vanguard' kernel tuning.
    """
    def __init__(self, threshold_ms: float = 50.0):
        self.threshold_ms = threshold_ms
        self.prometheus_url = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
        self.last_remediation = 0
        self.cooldown_period = 3600 # 1 hour between kernel tunings

    async def check_and_remediate(self):
        """
        Queries Prometheus for P95 latency and triggers kernel tuning if threshold exceeded.
        """
        import time
        if time.time() - self.last_remediation < self.cooldown_period:
            return

        try:
            p95_latency = await self._query_p95_latency()
            if p95_latency > self.threshold_ms:
                logger.warning(
                    "p95_latency_exceeded_threshold", 
                    latency=p95_latency, 
                    threshold=self.threshold_ms
                )
                await self._trigger_kernel_tuning()
                self.last_remediation = time.time()
        except Exception as e:
            logger.error("latency_remediation_failed", error=str(e))

    async def _query_p95_latency(self) -> float:
        """
        Executes PromQL to get the 95th percentile latency of the pricing service.
        """
        client = get_obs_client()
        query = 'histogram_quantile(0.95, sum(rate(pricing_service_duration_seconds_bucket[5m])) by (le))'
        
        try:
            resp = await client.get(
                f"{self.prometheus_url}/api/v1/query", 
                params={"query": query}
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract the scalar value from Prometheus result
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1]) * 1000 # convert to ms
            return 0.0
        except Exception:
            return 0.0

    async def _trigger_kernel_tuning(self):
        """
        Executes the 'Vanguard' optimize_kernel.sh script.
        Requires sudo/root permissions or pre-configured NOPASSWD in sudoers.
        """
        logger.info("triggering_os_kernel_reoptimization")
        # Optimization: In production, this would use an internal RPC or fabric/ansible
        # Here we demonstrate the loop by calling the script.
        proc = await asyncio.create_subprocess_exec(
            "sudo", "/app/scripts/optimize_kernel.sh",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("kernel_reoptimization_successful")
        else:
            logger.error("kernel_reoptimization_failed", error=stderr.decode())
