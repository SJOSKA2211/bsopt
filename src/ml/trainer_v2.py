import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any, Union
import logging
from pathlib import Path
import json
import time
import mlflow

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "models/checkpoints",
        scheduler: Optional[Any] = None,
        experiment_name: str = "Default_Experiment"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        
        # MLflow Init
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        self.log_params()

    def log_params(self):
        params = {
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "device": self.device,
            "model_class": self.model.__class__.__name__
        }
        mlflow.log_params(params)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        start_time = time.time()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Logging
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch
            }, step=epoch)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Checkpoint Best
            if val_loss == min(self.history["val_loss"]):
                self.save_checkpoint("best_model.pt")
                mlflow.log_artifact(str(self.output_dir / "best_model.pt"))

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.2f}s")
        mlflow.log_metric("total_time", total_time)
        self.save_metrics()
        mlflow.end_run()

    def save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def save_metrics(self):
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.history, f, indent=2)
        mlflow.log_artifact(str(metrics_path))
