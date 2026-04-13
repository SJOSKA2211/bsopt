import structlog

from src.database import health_check as db_health
from src.shared.utils.cache import get_redis_client

logger = structlog.get_logger(__name__)


async def check_redis() -> bool:
    """Verifies connection to the Redis caching layer."""
    try:
        redis = await get_redis_client()
        return await redis.ping()
    except Exception as e:
        logger.error("health_redis_check_failed", error=str(e))
        return False


def check_greeks_sanity() -> bool:
    """Verifies that the Greeks risk engine is operational."""
    try:
        # Simplified check: can we initialize a RiskAttributor?
        from src.portfolio.risk import RiskAttributor

        attributor = RiskAttributor([])
        attributor.aggregate_greeks()
        return True
    except Exception as e:
        logger.error("health_greeks_check_failed", error=str(e))
        return False


async def get_portfolio_health() -> dict:
    """Aggregates portfolio service health components."""
    # 1. DB Check (Existing)
    db_ok = db_health()

    # 2. Redis Check
    redis_ok = await check_redis()

    # 3. Greeks sanity Check
    greeks_ok = check_greeks_sanity()

    status = "healthy" if db_ok and redis_ok and greeks_ok else "degraded"

    return {
        "status": status,
        "database": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "disconnected",
        "risk_engine": "operational" if greeks_ok else "faulty",
        "service": "portfolio-service",
    }