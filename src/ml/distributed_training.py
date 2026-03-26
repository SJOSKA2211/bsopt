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

from src.config import settings
from src.ml.reinforcement_learning.decision_transformer import DecisionTransformer
from src.ml.reinforcement_learning.offline_train import TrajectoryDataset

logger = structlog.get_logger(__name__)

def train_func(config: dict[str, Any]):
    """
    Worker function for distributed Decision Transformer training.
    OPTIMIZED: torch.compile, AMP (GradScaler), and Grad Flow Monitoring.
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

    device = ray.train.torch.get_device()
    model = model.to(device)

    #  HIGH-PERFORMANCE: Kernel Fusion via torch.compile
    try:
        if config.get("use_compile", True):
            model = th.compile(model)
            logger.info("model_compiled_successfully")
    except Exception as e:
        logger.warning("torch_compile_failed", error=str(e))

    # Wrap for DDP
    model = ray.train.torch.prepare_model(model)

    optimizer = th.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-2),
        betas=(0.9, 0.95),  
    )
    criterion = nn.MSELoss()

    # ⚡ AMP: Automatic Mixed Precision
    use_amp = config.get("use_amp", th.cuda.is_available())
    scaler = th.cuda.amp.GradScaler(enabled=use_amp)

    # 3. Setup Data
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
        import json

        with open("data/trajectories.json", encoding="utf-8") as f:
            trajectories = json.load(f)
        dataset = TrajectoryDataset(trajectories)
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 64), shuffle=True)
        sharded_loader = ray.train.torch.prepare_data_loader(loader)

    # 4. Training Loop
    model.train()
    for epoch in range(config.get("epochs", 10)):
        epoch_loss = 0
        for batch in sharded_loader:
            optimizer.zero_grad()

            if isinstance(batch, dict):
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["rtg"].to(device)
                timesteps = batch["timesteps"].to(device)
            else:
                states, actions, rtg, timesteps = [x.to(device) for x in batch]

            # ⚡ AMP Forward Pass
            with th.cuda.amp.autocast(enabled=config.get("use_amp", True)):
                state_preds, action_preds, return_preds = model(states, actions, rtg, timesteps)

                # 1. Action Prediction Loss (P0)
                loss_action = criterion(action_preds, actions)

                # 2. Auxiliary Losses for Stability (P1)
                # Predict next state and next return
                loss_state = criterion(state_preds[:, :-1, :], states[:, 1:, :])
                loss_rtg = criterion(return_preds[:, :-1, :], rtg[:, 1:, :])

                loss = loss_action + 0.1 * loss_state + 0.1 * loss_rtg

            # ⚡ AMP Backward Pass
            scaler.scale(loss).backward()

            # Grad Clipping & Monitoring (Rank 0 only)
            if ray.train.get_context().get_local_rank() == 0:
                scaler.unscale_(optimizer)
                grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                mlflow.log_metric("grad_norm", grad_norm.item())

            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(sharded_loader)
        ray.train.report({"loss": avg_loss, "epoch": epoch})

        if ray.train.get_context().get_local_rank() == 0:
            mlflow.log_metric("dist_loss", avg_loss, step=epoch)
            # HIGH-PERFORMANCE: Log weight distribution and gradient flow
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    mlflow.log_metric(f"grad_norm_{name}", param.grad.norm().item(), step=epoch)
                if "weight" in name:
                    mlflow.log_metric(f"weight_mean_{name}", param.data.mean().item(), step=epoch)
                    mlflow.log_metric(f"weight_std_{name}", param.data.std().item(), step=epoch)

class BSOptDistributedTrainer:
    """
    Orchestrator for scaling BSOpt training across a Ray cluster.
    OPTIMIZED: Automatic resource negotiation and cluster-aware scaling.
    """

    def __init__(self, num_workers: int | None = None, use_gpu: bool | None = None):
        self._explicit_workers = num_workers
        self._explicit_gpu = use_gpu

    def _negotiate_resources(self) -> tuple[int, bool]:
        """Dynamically detect cluster capacity and set optimal worker count."""
        resources = ray.cluster_resources()
        cpus = int(resources.get("CPU", 1))
        gpus = int(resources.get("GPU", 0))

        # 1. Determine Worker Count
        if self._explicit_workers:
            num_workers = self._explicit_workers
        else:
            # Leave 1 CPU for the driver
            num_workers = max(1, cpus - 1)

        # 2. Determine GPU usage
        if self._explicit_gpu is not None:
            use_gpu = self._explicit_gpu
        else:
            # Use GPU if available and we have enough for workers
            use_gpu = gpus >= num_workers if num_workers > 0 else gpus > 0

        logger.info(
            "resource_negotiation_complete",
            cpus=cpus,
            gpus=gpus,
            workers=num_workers,
            use_gpu=use_gpu,
        )
        return num_workers, use_gpu

    def run(self, config: dict[str, Any]):
        """Starts the distributed training session using RayClusterManager."""

        if not HAS_RAY_TRAIN:
            logger.error("ray_train_missing")
            return None

        from src.shared.utils.ray_cluster_manager import RayClusterManager

        if not RayClusterManager.initialize():
            raise RuntimeError("Failed to initialize Ray cluster via RayClusterManager.")

        try:
            num_workers, use_gpu = self._negotiate_resources()

            #  HIGH-PERFORMANCE: Dynamic Scaling Config
            scaling_config = ScalingConfig(
                num_workers=num_workers,
                use_gpu=use_gpu,
                resources_per_worker={"CPU": 1, "GPU": 1 if use_gpu else 0},
            )

            trainer = TorchTrainer(
                train_func, train_loop_config=config, scaling_config=scaling_config
            )

            logger.info("starting_distributed_training", workers=num_workers, gpu=use_gpu)
            result = trainer.fit()
            return result
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
    parser.add_argument("--use_gpu", action="store_true")
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
        dt = BSOptDistributedTrainer(num_workers=args.workers, use_gpu=args.use_gpu)
        dt.run(config)
