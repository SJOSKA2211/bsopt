import structlog

from src.ingestion.engine import NSEScraper
from src.workers.tasks.celery_app import BaseTaskWithRetry, celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True, base=BaseTaskWithRetry, name="scrapers.refresh_nse_cache")
def refresh_nse_cache_task(self):
    """
    Celery task to refresh the NSE market data cache.
    Executed periodically to ensure data freshness.
    OPTIMIZED: Uses BaseAsyncTask for non-blocking execution.
    """
    logger.info("nse_refresh_task_triggered")
    scraper = NSEScraper()
    try:
        self.run_async(scraper._refresh_cache())
        logger.info("nse_refresh_task_success")
    except Exception as e:
        logger.error("nse_refresh_task_failed", error=str(e))
        raise e
    finally:
        # Clean shutdown (async)
        try:
            self.run_async(scraper.shutdown())
        except Exception:
            pass