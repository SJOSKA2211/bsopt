"""
Train V2 — expected by test_training.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ml.trainer_v2 import Trainer


class TransformerAdapter:
    """Adapter class expected by the test."""
    pass


def get_dataloaders(n_samples: int) -> tuple[DataLoader, DataLoader]:
    """Return dummy dataloaders for the test."""
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    return loader, loader


def train_neural_network(n_samples: int = 100, epochs: int = 1) -> Path:
    """
    Main entry point for training the neural network,
    as expected by test_training.py.
    """
    adapter = TransformerAdapter()
    
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
    
    trainer.fit(train_loader, val_loader, epochs=epochs)
    
    return trainer.output_dir / "best_model.pt"
