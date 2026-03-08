import asyncio

import structlog

from src.ml.celery_app import celery_app
from src.scrapers.engine import NSEScraper

logger = structlog.get_logger(__name__)


@celery_app.task(name="scrapers.refresh_nse_cache")
def refresh_nse_cache_task():
    """
    Celery task to refresh the NSE market data cache.
    Executed periodically to ensure data freshness.
    """
    logger.info("nse_refresh_task_triggered")
    scraper = NSEScraper()
    try:
        asyncio.run(scraper._refresh_cache())
        logger.info("nse_refresh_task_success")
    except Exception as e:
        logger.error("nse_refresh_task_failed", error=str(e))
        raise e
    finally:
        asyncio.run(scraper.shutdown())
