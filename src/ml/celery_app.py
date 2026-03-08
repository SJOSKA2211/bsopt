from typing import Any

import structlog
from celery import Celery

from src.config import get_settings
from src.ml.pipeline import MLPipeline

# Initialize structured logger
logger = structlog.get_logger()

settings = get_settings()

# Initialize Celery app
celery_app = Celery("bsopt_ml", broker=settings.RABBITMQ_URL, backend=settings.REDIS_URL)

celery_app.conf.update(
    task_serializer="msgpack",  # Optimized for speed
    result_serializer="msgpack",
    accept_content=["msgpack", "json"],
    timezone="UTC",
    enable_utc=True,
    # 🚀 Performance Tuning
    worker_prefetch_multiplier=1,  # Prevent long-running ML tasks from blocking others
    task_acks_late=True,           # Ensure reliability for ML pipelines
    worker_cancel_long_running_tasks_on_connection_loss=True,
    # Broker optimizations
    broker_pool_limit=10,
    redis_max_connections=20,
)


@celery_app.on_after_configure.connect
def setup_direct_queues(sender, **kwargs):
    """Register database cleanup on worker shutdown."""
    from celery.signals import worker_shutdown
    
    @worker_shutdown.connect
    def shutdown_db_manager(**kwargs):
        import asyncio
        from src.database import db_manager
        logger.info("celery_worker_shutting_down_cleaning_db_pool")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(db_manager.dispose())
            else:
                asyncio.run(db_manager.dispose())
        except Exception:
            pass


@celery_app.task(bind=True, name="ml.run_autonomous_pipeline")
def run_pipeline_task(self, config: dict[str, Any]):
    """
    Celery task to run the autonomous ML pipeline.
    """
    logger.info("celery_task_started", task_id=self.request.id, ticker=config.get("ticker"))
    pipeline = MLPipeline(config)
    try:
        import asyncio

        model = asyncio.run(pipeline.run())

        result = {
            "status": "success",
            "model_promoted": model is not None,
            "task_id": self.request.id,
        }

        logger.info("celery_task_completed", **result)
        return result

    except Exception as e:
        logger.error("celery_task_failed", error=str(e), task_id=self.request.id)
        # Re-raise so Celery marks it as failed
        raise e
    finally:
        import asyncio

        try:
            asyncio.run(pipeline.shutdown())
        except Exception:
            pass

# Import tasks to register them with the app
import src.scrapers.tasks # noqa
