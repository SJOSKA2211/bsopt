import asyncio
import os
import structlog
from sqlalchemy import text
from src.database import db_manager
from src.auth.vault_service import vault_service
from src.shared.utils.cache import get_redis_client
from src.shared.rabbitmq import get_rabbitmq

logger = structlog.get_logger(__name__)

async def check_database() -> bool:
    """Verifies connection to the Postgres backend."""
    try:
        async with db_manager.async_session_factory() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("health_db_check_failed", error=str(e))
        return False

def check_vault() -> bool:
    """Verifies connectivity and authentication with HashiCorp Vault."""
    try:
        return vault_service.is_authenticated()
    except Exception as e:
        logger.error("health_vault_check_failed", error=str(e))
        return False

async def check_redis() -> bool:
    """Verifies connection to the Redis cluster."""
    try:
        redis = await get_redis_client()
        return bool(await redis.ping())
    except Exception as e:
        logger.error("health_redis_check_failed", error=str(e))
        return False

async def check_rabbitmq() -> bool:
    """Verifies connectivity with RabbitMQ."""
    try:
        rmq = get_rabbitmq()
        if not rmq.connection or rmq.connection.is_closed:
            await rmq.connect()
        return not rmq.connection.is_closed
    except Exception as e:
        logger.error("health_rabbitmq_check_failed", error=str(e))
        return False

async def get_overall_health() -> dict:
    """Aggregates sub-service health into a single status report."""
    
    if os.environ.get("BYPASS_HEALTH_CHECK", "false").lower() == "true":
        return {
            "status": "healthy",
            "database": "simulated",
            "vault": "simulated",
            "redis": "simulated",
            "rabbitmq": "simulated",
            "service": "auth-service",
            "note": "Simulation mode active"
        }

    # Parallel Health Checks
    db_task = asyncio.create_task(check_database())
    redis_task = asyncio.create_task(check_redis())
    rmq_task = asyncio.create_task(check_rabbitmq())
    
    vault_ok = check_vault()
    db_ok = await db_task
    redis_ok = await redis_task
    rmq_ok = await rmq_task
    
    all_ok = all([db_ok, vault_ok, redis_ok, rmq_ok])
    status = "healthy" if all_ok else "degraded"
    
    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "vault": "active" if vault_ok else "inactive",
        "redis": "connected" if redis_ok else "disconnected",
        "rabbitmq": "connected" if rmq_ok else "disconnected",
        "service": "auth-service",
        "timestamp": os.popen("date -u +'%Y-%m-%dT%H:%M:%SZ'").read().strip()
    }
