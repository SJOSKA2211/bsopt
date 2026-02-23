from typing import Any

import ray
import ray.train.torch
import structlog
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from torch.utils.data import DataLoader, TensorDataset

try:
    from ray.train.torch import TorchTrainer

    HAS_RAY_TRAIN = True
except ImportError:
    TorchTrainer = None
    HAS_RAY_TRAIN = False

from src.config import settings
from src.ml.reinforcement_learning.decision_transformer import DecisionTransformer
from src.ml.reinforcement_learning.offline_train import TrajectoryDataset

logger = structlog.get_logger(__name__)


def train_func(config: dict[str, Any]):
    """
    Worker function for distributed Decision Transformer training.
    OPTIMIZED: Real model, real data, real tracking.
    """
    if not HAS_RAY_TRAIN:
        raise ImportError("Ray Train Torch dependencies missing.")

    # 1. Setup MLflow Tracking (Native Postgres)
    import mlflow

    mlflow.set_tracking_uri(settings.tracking_uri)

    # 2. Setup Model (Decision Transformer)
    model = DecisionTransformer(
        state_dim=config.get("state_dim", 100),
        action_dim=config.get("action_dim", 10),
        max_length=config.get("max_length", 20),
        max_ep_len=config.get("max_ep_len", 1000),
    )

    # Wrap for DDP
    model = ray.train.torch.prepare_model(model)

    optimizer = th.optim.AdamW(
        model.parameters(), lr=config.get("lr", 1e-4), weight_decay=1e-2
    )
    criterion = nn.MSELoss()

    # 3. Setup Data (Trajectory Loading)
    # Assume data is shared or reachable via NFS/Cloud Storage
    import pickle

    with open(config.get("dataset_path", "data/trajectories.pkl"), "rb") as f:
        trajectories = pickle.load(f)

    dataset = TrajectoryDataset(trajectories)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=True)
    sharded_loader = ray.train.torch.prepare_data_loader(loader)

    # 4. Training Loop
    model.train()
    for epoch in range(config.get("epochs", 10)):
        epoch_loss = 0
        for batch in sharded_loader:
            optimizer.zero_grad()

            _, action_preds, _ = model(
                batch["states"],
                th.zeros_like(batch["actions"]),
                batch["rtg"],
                batch["timesteps"],
            )

            loss = criterion(action_preds, batch["actions"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Report metrics to Ray Train
        avg_loss = epoch_loss / len(sharded_loader)
        ray.train.report({"loss": avg_loss, "epoch": epoch})

        if ray.train.get_context().get_local_rank() == 0:
            mlflow.log_metric("dist_loss", avg_loss, step=epoch)


class BSOptDistributedTrainer:
    """
    Orchestrator for scaling BSOpt training across a Ray cluster.
    """

    def __init__(self, num_workers: int = 2, use_gpu: bool = False):
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def run(self, config: dict[str, Any]):
        """Starts the distributed training session."""
        if not HAS_RAY_TRAIN:
            logger.error("ray_train_missing")
            return None

        # AUDIT: Ensure resources_per_trial is set for scalability
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
            resources_per_worker={"CPU": 1, "GPU": 1 if self.use_gpu else 0},
        )

        trainer = TorchTrainer(
            train_func, train_loop_config=config, scaling_config=scaling_config
        )

        logger.info("starting_distributed_training", workers=self.num_workers)
        result = trainer.fit()
        return result


if __name__ == "__main__":
    # Local verification if run directly
    ray.init(ignore_reinit_error=True)
    dt = BSOptDistributedTrainer(num_workers=1)  # 1 worker for local test
    dt.run({"lr": 1e-4, "epochs": 1, "dataset_size": 100})
