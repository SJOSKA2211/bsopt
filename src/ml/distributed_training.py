from typing import Any

import ray
import structlog
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from src.ml.trainer_v2 import Trainer

logger = structlog.get_logger(__name__)

def train_func(config: dict[str, Any]):
    """
    Worker function executed on each Ray node.
    """
    # 1. Setup Model, Optimizer, Criterion
    # For now using a simple placeholder, but in prod this uses architectures/
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    # 2. Wrap model for Distributed Data Parallel (DDP)
    model = ray.train.torch.prepare_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = nn.MSELoss()
    
    # 3. Setup Data (Sharded automatically by Ray Train)
    # train_loader = ... 
    # val_loader = ...
    # loader = ray.train.torch.prepare_data_loader(loader)
    
    # 4. Initialize the custom V2 Trainer
    # We pass the sharded loaders and the DDP model
    Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        experiment_name=config.get("experiment_name", "BSOpt_Distributed")
    )
    
    # trainer.fit(train_loader, val_loader, epochs=config.get("epochs", 10))
    logger.info("distributed_worker_ready", rank=ray.train.get_context().get_local_rank())

class BSOptDistributedTrainer:
    """
    Orchestrator for scaling BSOpt training across a cluster.
    """
    def __init__(self, num_workers: int = 2, use_gpu: bool = torch.cuda.is_available()):
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def run(self, config: dict[str, Any]):
        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=self.num_workers, use_gpu=self.use_gpu)
        )
        
        result = trainer.fit()
        logger.info("distributed_training_complete", metrics=result.metrics)
        return result

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    dt = BSOptDistributedTrainer()
    dt.run({"lr": 1e-4, "epochs": 5})
