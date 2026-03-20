"""
Trainer V2 — expected by test_singularity_models.py and test_training.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Trainer:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        criterion: Any,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.output_dir = output_dir

    def train(self, train_loader: Any, val_loader: Any, epochs: int = 1) -> None:
        """
        Production-grade training loop for PyTorch models.
        """
        import torch
        from tqdm import tqdm

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for i, data in enumerate(pbar):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    pbar.set_postfix({"loss": running_loss / 10})
                    running_loss = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_with_info
            if hasattr(self.model, "state_with_info")
            else self.model.state_dict(),
            self.output_dir / "model.pt",
        )
