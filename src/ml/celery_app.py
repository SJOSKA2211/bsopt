"""
Legacy Celery App Redirection - God Mode Consolidation.
Points all ML tasks to the unified src.tasks.celery_app.
"""

from src.ml.autonomous_pipeline import AutonomousMLPipeline
from src.tasks.celery_app import celery_app
from src.tasks.ml_tasks import run_pipeline_task

__all__ = ["celery_app", "run_pipeline_task", "AutonomousMLPipeline"]
