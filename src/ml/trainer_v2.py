import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "models/checkpoints",
        scheduler: Any | None = None,
        experiment_name: str = "Default_Experiment",
        is_distributed: bool = False
    ):
        self.is_distributed = is_distributed
        self.device = device
        
        if not self.is_distributed:
            self.model = model.to(device)
            # MLflow Init only for local runs
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
        else:
            # Distributed: Model assumed already wrapped/placed by orchestrator (e.g. Ray DDP)
            self.model = model
            self.run = None

        self.optimizer = optimizer
        self.criterion = criterion
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}
        
        if not self.is_distributed:
            # Log params only after attributes are set
            self._log_params()

    def _log_params(self):
        """Logs model and optimizer parameters to MLflow."""
        params = {
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "device": self.device,
            "model_class": self.model.__class__.__name__
        }
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

    def validate(self, loader: DataLoader) -> float:
        """Validates the model."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(loader)

    def _handle_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Processes logic at the end of each epoch."""
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        # Logging
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        }
        
        if self.is_distributed and HAS_RAY:
            ray.train.report(metrics)
            
        # Only log to MLflow/Console if local or Rank 0
        should_log = True
        if self.is_distributed and HAS_RAY:
            context = ray.train.get_context()
            if context.get_local_rank() != 0:
                should_log = False

        if should_log:
            if self.run: # Check if MLflow run exists
                mlflow.log_metrics(metrics, step=epoch)
            
            logger.info(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Checkpoint Best
            if val_loss == min(self.history["val_loss"]):
                self._save_checkpoint("best_model.pt")
                if self.run:
                    mlflow.log_artifact(str(self.output_dir / "best_model.pt"))

        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
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
            val_loss = self.validate(val_loader)
            
            self._handle_epoch_end(epoch, train_loss, val_loss)

            early_stopping(val_loss)
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
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def _save_metrics(self):
        """Saves history to JSON and logs as artifact."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.history, f, indent=2)
        mlflow.log_artifact(str(metrics_path))
