from typing import Any

import ray
import ray.train.torch
import structlog
import torch as th
import torch.nn as nn
from ray.train import ScalingConfig
from torch.utils.data import DataLoader

try:
    from ray.train.torch import TorchTrainer

    HAS_RAY_TRAIN = True
except ImportError:
    TorchTrainer = None
    HAS_RAY_TRAIN = False

from src.shared.config import settings
from src.ml.reinforcement_learning.decision_transformer import DecisionTransformer
from src.ml.reinforcement_learning.offline_train import TrajectoryDataset

logger = structlog.get_logger(__name__)


def train_func(config: dict[str, Any]):
    """
    Worker function for distributed Decision Transformer training.
    Optimized for pure CPU execution.
    """
    if not HAS_RAY_TRAIN:
        raise ImportError("Ray Train Torch dependencies missing.")

    # 1. Setup MLflow Tracking
    import mlflow

    mlflow.set_tracking_uri(settings.tracking_uri)

    # 2. Setup Model (DT-v2)
    model = DecisionTransformer(
        state_dim=config.get("state_dim", 128),
        action_dim=config.get("action_dim", 10),
        max_length=config.get("max_length", 20),
        max_ep_len=config.get("max_ep_len", 1000),
    )

    import ray
device = th.device("cpu")
model = model.to(device)

# Wrap for DDP (using 'gloo' backend for CPU)
model = ray.train.torch.prepare_model(model)

optimizer = th.optim.AdamW(
...
optimizer = th.optim.AdamW(
    model.parameters(),
    lr=config.get("lr", 1e-4),
    weight_decay=config.get("weight_decay", 1e-2),
    betas=(0.9, 0.95),
)
criterion = nn.MSELoss()

# Setup Data
import ray.data
    dataset_path = config.get("dataset_path", "data/trajectories.parquet")

    try:
        #  HIGH-PERFORMANCE: Streaming sharded data loading
        if dataset_path.endswith(".parquet"):
            ds = ray.data.read_parquet(dataset_path)
        else:
            # Fallback to JSON if specified
            ds = ray.data.read_json(dataset_path)

        # Create an iterator that shards the data across Ray workers automatically
        sharded_loader = ds.iter_torch_batches(
            batch_size=config.get("batch_size", 64), local_shuffle_buffer_size=1000
        )
        logger.info("sharded_loader_optimized", path=dataset_path)
    except Exception as e:
        logger.warning("ray_data_fallback_to_local", error=str(e))
        # Fallback to local loading if Ray Data fails
        import pickle  # nosec B403

        with open("data/trajectories.pkl", "rb") as f:
            trajectories = pickle.load(f)  # nosec B301
        dataset = TrajectoryDataset(trajectories)
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=True)
        sharded_loader = ray.train.torch.prepare_data_loader(loader)

    # 4. Training Loop (Pure CPU)
    model.train()
    for epoch in range(config.get("epochs", 10)):
        epoch_loss = 0
        for batch in sharded_loader:
            optimizer.zero_grad()
            
            states, actions, rtg, timesteps = (
                (batch["states"], batch["actions"], batch["rtg"], batch["timesteps"])
                if isinstance(batch, dict) else batch
            )
            states, actions, rtg, timesteps = [x.to(device) for x in [states, actions, rtg, timesteps]]

            state_preds, action_preds, return_preds = model(states, actions, rtg, timesteps)

            # Combined Loss
            loss = criterion(action_preds, actions) + \
                   0.1 * criterion(state_preds[:, :-1, :], states[:, 1:, :]) + \
                   0.1 * criterion(return_preds[:, :-1, :], rtg[:, 1:, :])

            loss.backward()

            if ray.train.get_context().get_local_rank() == 0:
                th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(sharded_loader)
        ray.train.report({"loss": avg_loss, "epoch": epoch})
        
        if ray.train.get_context().get_local_rank() == 0:
            mlflow.log_metric("dist_loss", avg_loss, step=epoch)


class BSOptDistributedTrainer:
    """Optimized Orchestrator for scaling BSOpt training natively on CPU."""
    def __init__(self, num_workers: int | None = None):
        self._explicit_workers = num_workers

    def _negotiate_resources(self) -> int:
        """Dynamically detect CPU capacity for worker scaling."""
        resources = ray.cluster_resources()
        cpus = int(resources.get("CPU", 1))
        return self._explicit_workers if self._explicit_workers else max(1, cpus - 1)

    def run(self, config: dict[str, Any]):
        if not HAS_RAY_TRAIN:
            logger.error("ray_train_missing")
            return None

        from src.shared.utils.ray_cluster_manager import RayClusterManager
        if not RayClusterManager.initialize():
            raise RuntimeError("Ray Cluster Initialization Failed")

        try:
            num_workers = self._negotiate_resources()
            scaling_config = ScalingConfig(num_workers=num_workers, resources_per_worker={"CPU": 1})
            trainer = TorchTrainer(train_func, train_loop_config=config, scaling_config=scaling_config)
            
            logger.info("starting_distributed_training_cpu", workers=num_workers)
            return trainer.fit()
        finally:
            if settings.RAY_SHUTDOWN_AFTER_RUN:
                RayClusterManager.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Distributed DT Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="data/trajectories.parquet")
    parser.add_argument("--study_name", type=str, default="distributed_dt_v1")
    parser.add_argument("--tracking_uri", type=str, default=None)

    args = parser.parse_args()

    if args.tracking_uri:
        # settings is immutable, but we can bypass it if needed or just use args.tracking_uri
        pass

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    config = {
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_path": args.dataset,
        "study_name": args.study_name,
    }

    from src.ml.tracker import ExperimentTracker

    tracker = ExperimentTracker(args.study_name, tracking_uri=args.tracking_uri)

    with tracker.start_run(nested=True):
        dt.run(config)