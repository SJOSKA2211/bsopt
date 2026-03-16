"""
Unified Base Trainer for BS-OPT
Provides shared functionality for MLflow tracking, logging, and experiment management.
"""

from abc import ABC, abstractmethod
from typing import Any

import mlflow
import structlog

from services.config import get_settings
from services.shared.observability import setup_logging

logger = structlog.get_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract Base Class for all Model Trainers.
    """

    def __init__(self, study_name: str, tracking_uri: str | None = None) -> None:
        setup_logging()
        self.settings = get_settings()
        self.study_name = study_name
        self.tracking_uri = tracking_uri or self.settings.tracking_uri

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.study_name)

    @abstractmethod
    def train_and_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Execute training and return evaluation metric."""
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs parameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Sets tags for the current MLflow run."""
        mlflow.set_tags(tags)
