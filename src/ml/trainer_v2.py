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
        Mock training loop.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model.pt").touch()
