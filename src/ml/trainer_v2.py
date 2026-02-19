import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import ray.train

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from src.ml.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """
    Optimized Trainer for Neural Networks.
    Handles training loop, validation, checkpointing, and MLflow logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | None = None,
        output_dir: str = "models/checkpoints",
        scheduler: Any | None = None,
        experiment_name: str = "Default_Experiment",
        is_distributed: bool = False,
    ):
        self.is_distributed = is_distributed
        self.rank = 0

        if self.is_distributed and HAS_RAY:
            try:
                self.rank = ray.train.get_context().get_local_rank()
                # Auto-detect device for Ray workers
                self.device = (
                    f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
                )
            except (ImportError, RuntimeError):
                self.rank = 0
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.is_distributed:
            self.model = model.to(self.device)
            # MLflow Init only for local runs
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
        else:
            # Distributed: Model assumed already wrapped/placed by orchestrator (e.g. Ray DDP)
            self.model = model
            # Rank 0 in distributed mode also initializes MLflow for centralized tracking
            if self.rank == 0:
                mlflow.set_experiment(experiment_name)
                self.run = mlflow.start_run(nested=True)
            else:
                self.run = None

        self.optimizer = optimizer
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

        if self.rank == 0:
            # Log params only after attributes are set
            self._log_params()

    def _log_params(self):
        """Logs model and optimizer parameters to MLflow."""
        params = {
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "device": self.device,
            "model_class": self.model.__class__.__name__,
            "is_distributed": self.is_distributed,
        }
        if self.run:
            mlflow.log_params(params)

    def train_epoch(self, loader: DataLoader) -> float:
        """Trains for one epoch."""
        self.model.train()
        total_loss = 0.0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """
        Validates the model and calculates comprehensive metrics using ModelScorecard.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        from src.ml.evaluation.metrics import ModelScorecard

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_loss = total_loss / len(loader)

        # Concatenate all batches for full evaluation
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)

        # Calculate returns if target is 'log_return' (Simplified assumption for now)
        # In production, we'd pass 'returns' explicitly to the score card.
        scorecard = ModelScorecard(y_true, y_pred)
        metrics = scorecard.to_dict()
        metrics["loss"] = avg_loss

        return metrics

    def _handle_epoch_end(self, epoch: int, metrics: dict[str, float]):
        """Processes logic at the end of each epoch."""
        # Sync metrics across all workers in distributed mode
        from src.ml.utils.distributed import sync_metrics

        synced_metrics = sync_metrics(metrics)

        self.history["train_loss"].append(synced_metrics.get("train_loss", 0.0))
        self.history["val_loss"].append(synced_metrics["loss"])

        if self.is_distributed and HAS_RAY:
            ray.train.report(synced_metrics)

        # Only log to MLflow/Console if local or Rank 0
        should_log = True
        if self.is_distributed and HAS_RAY:
            context = ray.train.get_context()
            if context.get_local_rank() != 0:
                should_log = False

        if should_log:
            if self.run:  # Check if MLflow run exists
                mlflow.log_metrics(synced_metrics, step=epoch)

            logger.info(
                f"Epoch {epoch+1} | "
                f"Train Loss: {synced_metrics.get('train_loss', 0.0):.4f} | "
                f"Val Loss: {synced_metrics['loss']:.4f} | "
                f"RMSE: {synced_metrics.get('rmse', 0.0):.4f} | "
                f"R2: {synced_metrics.get('r2', 0.0):.4f}"
            )

            # Checkpoint Best based on composite score or loss
            if synced_metrics["loss"] == min(self.history["val_loss"]):
                self._save_checkpoint("best_model.pt")
                if self.run:
                    mlflow.log_artifact(str(self.output_dir / "best_model.pt"))

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(synced_metrics["loss"])
            else:
                self.scheduler.step()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        """Main training entry point."""
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        start_time = time.time()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_metrics["train_loss"] = train_loss

            self._handle_epoch_end(epoch, val_metrics)

            early_stopping(val_metrics["loss"])
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.2f}s")

        if self.run:
            mlflow.log_metric("total_time", total_time)
            self._save_metrics()
            mlflow.end_run()

    def _save_checkpoint(self, filename: str):
        """Saves a model checkpoint."""
        path = self.output_dir / filename

        # Unwrap DDP model if necessary
        model_state = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def _save_metrics(self):
        """Saves history to JSON and logs as artifact (Rank 0 only)."""
        if self.rank != 0:
            return

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.history, f, indent=2)
        if self.run:
            mlflow.log_artifact(str(metrics_path))
