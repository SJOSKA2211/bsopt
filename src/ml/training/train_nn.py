import logging
import os

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from src.ml.architectures.neural_network import OptionPricingNN
from src.ml.training.train import load_or_collect_data

logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")  # or "gloo" for CPU
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


async def train_nn_distributed(epochs: int = 10, batch_size: int = 64):
    """
    Distributed training for the Option Pricing NN.
    """
    rank, world_size = setup_distributed()
    is_main = rank == 0

    # Load data (only on main or broadcast)
    X, y, feature_names, _ = await load_or_collect_data(use_real_data=False, n_samples=20000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None)
    )

    # Model
    model = OptionPricingNN(input_dim=len(feature_names))
    if world_size > 1:
        model = DDP(model.to(rank), device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if is_main:
        mlflow.set_experiment("Option_Pricing_NN")
        mlflow.start_run()

    for epoch in range(epochs):
        if sampler:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if is_main:
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            logger.info(f"Epoch {epoch}: Loss {avg_loss:.4f}")

    if is_main:
        # Save model and export ONNX
        torch.save(model.state_dict(), "models/nn_option_pricer.pth")

        # Export ONNX for efficient inference
        sample_input = X_tensor[:1]
        (
            model.module.export_onnx("models/nn_option_pricer.onnx", sample_input)
            if world_size > 1
            else model.export_onnx("models/nn_option_pricer.onnx", sample_input)
        )

        mlflow.log_artifact("models/nn_option_pricer.onnx")
        mlflow.end_run()


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(train_nn_distributed())
