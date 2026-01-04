from celery import Celery
from typing import Dict, Any
from src.ml.autonomous_pipeline import AutonomousMLPipeline
import structlog

# Initialize structured logger
logger = structlog.get_logger()

# Initialize Celery app
# In production, these should be loaded from env vars
BROKER_URL = "redis://localhost:6379/0"
BACKEND_URL = "redis://localhost:6379/0"

celery_app = Celery(
    "bsopt_ml",
    broker=BROKER_URL,
    backend=BACKEND_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(bind=True, name="ml.run_autonomous_pipeline")
def run_pipeline_task(self, config: Dict[str, Any]):
    """
    Celery task to run the autonomous ML pipeline.
    """
    logger.info("celery_task_started", task_id=self.request.id, ticker=config.get("ticker"))
    try:
        pipeline = AutonomousMLPipeline(config)
        study = pipeline.run()
        
        result = {
            "status": "success",
            "best_value": study.best_value,
            "best_params": study.best_params,
            "task_id": self.request.id
        }
        logger.info("celery_task_completed", **result)
        return result
        
    except Exception as e:
        logger.error("celery_task_failed", error=str(e), task_id=self.request.id)
        # Re-raise so Celery marks it as failed
        raise e
