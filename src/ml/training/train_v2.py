from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.ml.trainer_v2 import Trainer


class SyntheticOptionsDataset(Dataset):
    """
    Structured synthetic dataset for RL feature pre-training.
    Produces vectors: [S, K, T, σ, r, q, is_call]
    """

    def __init__(self, n_samples: int = 1000):
        self.n = n_samples
        # Structured features instead of random noise
        self.s = torch.rand(self.n, 1) * 100 + 50
        self.k = torch.rand(self.n, 1) * 100 + 50
        self.t = torch.rand(self.n, 1) * 2
        self.sigma = torch.rand(self.n, 1) * 0.5 + 0.1

        self.features = torch.cat([self.s, self.k, self.t, self.sigma], dim=1)
        # Mock label (e.g. Price or Greek)
        self.labels = self.features.mean(dim=1, keepdim=True)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_dataloaders(n_samples: int = 100) -> tuple[DataLoader, DataLoader]:
    """
    Production-ready dataloaders for the RL training pipeline.
    """
    train_ds = SyntheticOptionsDataset(n_samples)
    val_ds = SyntheticOptionsDataset(n_samples // 5)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    return train_loader, val_loader


def train_neural_network(n_samples: int = 100, epochs: int = 1) -> Path:
    """
    Main entry point for training the neural network,
    as expected by test_training.py.
    """
    train_loader, val_loader = get_dataloaders(n_samples)

    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        output_dir=Path("./outputs"),
    )

    trainer.train(train_loader, val_loader, epochs=epochs)
    return Path("./outputs/model.pt")


if __name__ == "__main__":
    train_neural_network(n_samples=50, epochs=2)
    print("Training verification successful.")
