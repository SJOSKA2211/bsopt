from typing import Any


class EarlyStopping:
    """Simple early stopping callback."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class MLflowCallback:
    """
    God-Mode: MLflow logging callback for custom training loops.
    """

    def __init__(self, run_name: str | None = None):
        import mlflow

        self.mlflow = mlflow
        self.run_name = run_name

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]):
        self.mlflow.log_metrics(metrics, step=epoch)


class ModelCheckpoint:
    """
    Automated model checkpointing with stage promotion support.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else -float("inf")

    def __call__(self, current_score: float, model: Any):
        import os

        import torch

        is_best = (
            (current_score < self.best_score)
            if self.mode == "min"
            else (current_score > self.best_score)
        )

        if is_best:
            self.best_score = current_score
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            return True
        return False
