import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.ml.trainer_v2 import Trainer
from src.config import settings
from typing import Tuple
import logging

# Fallback or Import
try:
    from src.ml.architectures.neural_network import NeuralNetwork
except ImportError:
    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.net(x)

logger = logging.getLogger(__name__)

def get_dataloaders(n_samples=1000) -> Tuple[DataLoader, DataLoader]:
    # Placeholder for real data loading logic
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)

def train_neural_network(n_samples: int = 10000, epochs: int = 10):
    logger.info("Starting Neural Network Training V2")
    
    train_loader, val_loader = get_dataloaders(n_samples)
    
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        experiment_name="BSOpt_Neural_Greeks"
    )
    
    trainer.fit(train_loader, val_loader, epochs=epochs)
    return trainer.output_dir / "best_model.pt"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_neural_network()
