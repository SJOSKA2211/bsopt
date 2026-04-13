import asyncio

import httpx
import structlog

logger = structlog.get_logger(__name__)

SERVICES = {
    "Auth": "http://auth-service:3001/health",
    "API": "http://api:8000/health",
    "ML": "http://ml-inference:5001/health",
    "Pricing": "http://neural-pricing:8000/health",
}


class SystemSentinel:
    """
    Production System Sentinel.
    Aggregates health status and latency across all microservices.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=2.0)

    async def check_health(self) -> dict[str, str]:
        """Aggregate health status from all services."""
        results = {}
        for name, url in SERVICES.items():
            try:
                resp = await self.client.get(url)
                results[name] = "HEALTHY" if resp.status_code == 200 else "DEGRADED"
            except Exception:
                results[name] = "DOWN"

        return results

    async def sentinel_loop(self):
        """Continuous monitoring and alerting."""
        logger.info("system_sentinel_started")
        while True:
            health = await self.check_health()
            if any(status != "HEALTHY" for status in health.values()):
                logger.warning("system_degradation_detected", status=health)
                import os

                slack_url = os.getenv("SLACK_WEBHOOK_URL")
                if slack_url:
                    try:
                        alert = {"text": f" *System Degradation Detected*\n```{health}```"}
                        await self.client.post(slack_url, json=alert)
                    except Exception as e:
                        logger.error("failed_to_send_slack_alert", error=str(e))

            await asyncio.sleep(60)


if __name__ == "__main__":
    sentinel = SystemSentinel()
    asyncio.run(sentinel.sentinel_loop())
