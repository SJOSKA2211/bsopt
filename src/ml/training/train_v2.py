"""
Machine Learning Training V2 (Optimized)
"""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.ml.reinforcement_learning.transformer_policy import DecisionTransformer
from src.ml.trainer_v2 import Trainer
from src.ml.utils.validation import WalkForwardValidator

logger = logging.getLogger(__name__)

class TransformerAdapter(nn.Module):
    """
    Adapts the DecisionTransformer for Supervised Learning loops.
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.transformer = DecisionTransformer(
            state_dim=input_dim,
            act_dim=output_dim,
            hidden_size=hidden_dim,
            max_length=1,
            max_ep_len=4096
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        states = x.unsqueeze(1)
        actions = torch.zeros(batch_size, 1, 1, device=device)
        returns = torch.zeros(batch_size, 1, 1, device=device)
        timesteps = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        _, action_preds, _ = self.transformer(states, actions, returns, timesteps)
        return action_preds.squeeze(1)

def get_real_dataset(n_samples=5000):
    """Simulated real data for validation loop testing."""
    X = torch.randn(n_samples, 10)
    # Target is some non-linear combination of input
    y = (X[:, 0] * X[:, 1] + torch.sin(X[:, 2])).unsqueeze(1)
    return TensorDataset(X, y)

def train_with_cross_validation(n_samples: int = 10000, epochs: int = 5):
    logger.info("Starting MLOps Hardened Training (Walk-Forward CV)")
    
    dataset = get_real_dataset(n_samples)
    validator = WalkForwardValidator(n_splits=3)
    
    # We need numpy indices for the validator
    X_indices = np.arange(len(dataset))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(validator.split(X_indices)):
        logger.info(f"Training Fold {fold+1}/{validator.get_n_splits()}")
        
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=64)
        
        model = TransformerAdapter(input_dim=10, hidden_dim=64, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            experiment_name=f"BSOpt_MLOps_Fold_{fold+1}",
            output_dir=f"models/checkpoints/fold_{fold+1}"
        )
        
        trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Capture the best val loss for this fold
        best_val = min(trainer.history["val_loss"])
        fold_results.append(best_val)
        logger.info(f"Fold {fold+1} Best Val Loss: {best_val:.4f}")

    avg_performance = np.mean(fold_results)
    logger.info(f"Cross-Validation Complete. Average Val Loss: {avg_performance:.4f}")
    return avg_performance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_with_cross_validation()
