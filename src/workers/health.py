import asyncio
import os
import structlog
import ray
from src.workers.tasks.celery_app import celery_app
from src.shared.config import settings

logger = structlog.get_logger(__name__)

async def check_celery_broker() -> bool:
    """Verifies connectivity to the Celery message broker (RabbitMQ)."""
    try:
        # Ping the broker
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=1)
        return True
    except Exception as e:
        logger.error("celery_broker_check_failed", error=str(e))
        return False

async def check_celery_workers() -> bool:
    """Verifies that at least one worker is alive and responding."""
    try:
        # inspect().ping() returns a dict of worker responses
        i = celery_app.control.inspect()
        pings = i.ping()
        return pings is not None and len(pings) > 0
    except Exception as e:
        logger.error("celery_worker_ping_failed", error=str(e))
        return False

async def check_ray_health() -> bool:
    """Verifies that Ray is initialized and the head node is reachable."""
    try:
        if not ray.is_initialized():
            # In a container environment, we'd typically connect to an existing cluster
            ray_address = os.getenv("RAY_ADDRESS", "auto")
            ray.init(address=ray_address, ignore_reinit_error=True)
        
        # Check node status
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n["Alive"]]
        return len(alive_nodes) > 0
    except Exception as e:
        logger.error("ray_health_check_failed", error=str(e))
        return False

async def get_worker_health() -> dict:
    """Aggregates all worker-related health components."""
    celery_ok = await check_celery_broker()
    workers_ok = await check_celery_workers()
    ray_ok = await check_ray_health()
    
    status = "healthy" if celery_ok and workers_ok and ray_ok else "degraded"
    
    return {
        "status": status,
        "celery_broker": "connected" if celery_ok else "disconnected",
        "workers_alive": workers_ok,
        "ray_cluster": "connected" if ray_ok else "disconnected",
        "service": "worker-development-cluster"
    }
