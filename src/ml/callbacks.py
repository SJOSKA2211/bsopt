import os
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


class BaseCallback:
    """Production-grade base class for all ML training hooks."""

    def on_train_begin(self, params: dict[str, Any]) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_batch_begin(self, batch: int) -> None:
        pass

    def on_batch_end(self, batch: int, logs: dict[str, Any]) -> None:
        pass


class EarlyStopping(BaseCallback):
    """Robust early stopping hook with patience and delta steering."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("early_stopping_triggered", epoch=epoch, best_loss=self.best_loss)

    def __call__(self, val_loss: float) -> None:
        """Convenience method to trigger early stopping check."""
        self.on_epoch_end(0, {"val_loss": val_loss})


class MLflowCallback(BaseCallback):
    """Real-time MLflow telemetry hook."""

    def __init__(self, run_name: str | None = None) -> None:
        import mlflow

        self.mlflow = mlflow
        self.run_name = run_name

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        self.mlflow.log_metrics(metrics, step=epoch)


class ModelCheckpoint(BaseCallback):
    """
    Automated high-fidelity model persistence hook.
    Ensures atomic writes and stage promotion support.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min") -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else -float("inf")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], model: Any = None) -> None:
        if model is None:
            return

        current_score = metrics.get(self.monitor)
        if current_score is None:
            return

        is_best = (
            (current_score < self.best_score)
            if self.mode == "min"
            else (current_score > self.best_score)
        )

        if is_best:
            self.best_score = current_score
            self._save_atomic(model, epoch)

    def _save_atomic(self, model: Any, epoch: int) -> None:
        """Saves model state using an atomic rename to prevent corruption."""
        temp_path = f"{self.filepath}.tmp"
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_score": self.best_score,
            },
            temp_path,
        )

        os.replace(temp_path, self.filepath)
        logger.info("model_checkpoint_saved", path=self.filepath, score=self.best_score)