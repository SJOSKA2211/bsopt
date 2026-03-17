"""
Legacy Celery App Redirection - High-Performance Consolidation.
Points all ML tasks to the unified services.workers.tasks.celery_app.
"""

from services.ml.autonomous_pipeline import AutonomousMLPipeline
from services.workers.tasks.celery_app import celery_app
from services.workers.tasks.ml_tasks import run_pipeline_task

__all__ = ["celery_app", "run_pipeline_task", "AutonomousMLPipeline"]
