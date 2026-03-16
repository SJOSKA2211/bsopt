"""
Legacy Celery App Redirection - High-Performance Consolidation.
Points all ML tasks to the unified src.tasks.celery_app.
"""

from services.ml.autonomous_pipeline import AutonomousMLPipeline
from services.tasks.celery_app import celery_app
from services.tasks.ml_tasks import run_pipeline_task

__all__ = ["celery_app", "run_pipeline_task", "AutonomousMLPipeline"]
