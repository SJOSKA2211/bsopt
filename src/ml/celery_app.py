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
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


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
