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
        state_dim=config.get("state_dim", 100),
        action_dim=config.get("action_dim", 10),
        max_length=config.get("max_length", 20),
        max_ep_len=config.get("max_ep_len", 1000),
    )

    device = ray.train.torch.get_device()
    model = model.to(device)

    # 🚀 GOD-MODE: Kernel Fusion via torch.compile
    try:
        if config.get("use_compile", True):
            model = th.compile(model)
            logger.info("model_compiled_successfully")
    except Exception as e:
        logger.warning("torch_compile_failed", error=str(e))

    # Wrap for DDP
    model = ray.train.torch.prepare_model(model)

    optimizer = th.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=1e-2)
    criterion = nn.MSELoss()

    # ⚡ AMP: Automatic Mixed Precision
    scaler = th.cuda.amp.GradScaler(enabled=config.get("use_amp", True))

    # 3. Setup Data
    import ray.data
    dataset_path = config.get("dataset_path", "data/trajectories.pkl")
    # ... (rest of data loading logic remains the same)
    ds = None
    try:
        if dataset_path.endswith(".json"):
            ds = ray.data.read_json(dataset_path)
    except Exception:
        pass

    if ds:
        sharded_loader = ds.iter_torch_batches(batch_size=config.get("batch_size", 64))
    else:
        import pickle
        with open(dataset_path, "rb") as f:
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

            if isinstance(batch, dict):
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["rtg"].to(device)
                timesteps = batch["timesteps"].to(device)
            else:
                states, actions, rtg, timesteps = [x.to(device) for x in batch]

            # ⚡ AMP Forward Pass
            with th.cuda.amp.autocast(enabled=config.get("use_amp", True)):
                action_preds = model(states, th.zeros_like(actions), rtg, timesteps)
                loss = criterion(action_preds, actions)

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
            # Log weight distribution
            for name, param in model.named_parameters():
                if "weight" in name:
                    mlflow.log_metric(f"weight_mean_{name}", param.data.mean().item())



class BSOptDistributedTrainer:
    """
    Orchestrator for scaling BSOpt training across a Ray cluster.
    """

    def __init__(self, num_workers: int = 2, use_gpu: bool = False):
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def run(self, config: dict[str, Any]):
        """Starts the distributed training session using RayClusterManager."""
        if not HAS_RAY_TRAIN:
            logger.error("ray_train_missing")
            return None

        from src.utils.ray_cluster_manager import RayClusterManager
        if not RayClusterManager.initialize():
            raise RuntimeError("Failed to initialize Ray cluster via RayClusterManager.")

        try:
            # AUDIT: Ensure resources_per_trial is set for scalability
            scaling_config = ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
                resources_per_worker={"CPU": 1, "GPU": 1 if self.use_gpu else 0},
            )

            trainer = TorchTrainer(train_func, train_loop_config=config, scaling_config=scaling_config)

            logger.info("starting_distributed_training", workers=self.num_workers)
            result = trainer.fit()
            return result
        finally:
            if os.getenv("RAY_SHUTDOWN_AFTER_RUN") == "true":
                RayClusterManager.shutdown()


if __name__ == "__main__":
    # Local verification if run directly
    ray.init(ignore_reinit_error=True)
    dt = BSOptDistributedTrainer(num_workers=1)  # 1 worker for local test
    dt.run({"lr": 1e-4, "epochs": 1, "dataset_size": 100})
