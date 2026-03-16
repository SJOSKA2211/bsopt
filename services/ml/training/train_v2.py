from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from services.ml.trainer_v2 import Trainer


def get_dataloaders(n_samples: int = 100) -> tuple[Any, Any]:
    """
    Mock dataloaders.
    """
    loader = [torch.randn(10, 10) for _ in range(n_samples)]
    return loader, loader


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
