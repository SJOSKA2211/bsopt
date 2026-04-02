import asyncio
import os
import structlog
from sqlalchemy import text
from src.database import db_manager
from src.auth.vault_service import vault_service

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

async def get_overall_health() -> dict:
    """Aggregates sub-service health into a single status report."""
    
    if os.environ.get("BYPASS_HEALTH_CHECK", "false").lower() == "true":
        return {
            "status": "healthy",
            "database": "simulated",
            "vault": "simulated",
            "service": "auth-service",
            "note": "Simulation mode active"
        }

    # 1. DB Check
    db_ok = await check_database()
    
    # 2. Vault Check
    vault_ok = check_vault()
    
    status = "healthy" if db_ok and vault_ok else "degraded"
    
    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "vault": "active" if vault_ok else "inactive",
        "service": "auth-service"
    }
