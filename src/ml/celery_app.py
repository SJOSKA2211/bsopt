"""
Legacy Celery App Redirection - High-Performance Consolidation.
Points all ML tasks to the unified src.workers.tasks.celery_app.
"""

from src.ml.autonomous_pipeline import AutonomousMLPipeline
from src.workers.tasks.celery_app import celery_app
from src.workers.tasks.ml_tasks import run_pipeline_task

__all__ = ["celery_app", "run_pipeline_task", "AutonomousMLPipeline"]