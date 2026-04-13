"""
Unified Base Trainer for BS-OPT
Provides shared functionality for MLflow tracking, logging, and experiment management.
"""

from abc import ABC, abstractmethod

import mlflow
import msgspec
import numpy as np
import structlog

from src.config import get_settings
from src.shared.observability import setup_logging

logger = structlog.get_logger(__name__)


class TrainingConfig(msgspec.Struct):
    framework: str = "xgboost"
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001
    metadata: dict[str, str] = {}


class TrainingResult(msgspec.Struct):
    score: float
    model_path: str | None = None
    metadata: dict[str, str] = {}


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
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        metadata: dict[str, str] | None = None,
    ) -> TrainingResult:
        """Execute training and return standardized result."""
        pass

    def log_params(self, params: dict[str, str | int | float | bool]) -> None:
        """Logs parameters to MLflow."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Sets tags for the current MLflow run."""
        mlflow.set_tags(tags)