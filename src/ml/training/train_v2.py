import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.reinforcement_learning.transformer_policy import DecisionTransformer
from src.ml.trainer_v2 import Trainer

logger = logging.getLogger(__name__)

class TransformerAdapter(nn.Module):
    """
    Adapts the DecisionTransformer for Supervised Learning loops.
    Treats the input X as 'state' and predicts 'action'.
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
        # x: (batch, input_dim)
        batch_size = x.shape[0]
        device = x.device
        
        # Reshape for Sequence format: (batch, seq_len=1, dim)
        states = x.unsqueeze(1)
        
        # Dummy inputs for other modalities (Zero-shot / BC style)
        actions = torch.zeros(batch_size, 1, 1, device=device)
        returns = torch.zeros(batch_size, 1, 1, device=device)
        timesteps = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Forward pass
        state_preds, action_preds, return_preds = self.transformer(
            states, actions, returns, timesteps
        )
        
        # Return action prediction (squeezed) as the model output
        return action_preds.squeeze(1)

def get_dataloaders(n_samples=1000) -> tuple[DataLoader, DataLoader]:
    # Placeholder for real data loading logic
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 1)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)

def train_neural_network(n_samples: int = 10000, epochs: int = 10):
    logger.info("Starting Neural Network Training V2 (Transformer Policy)")
    
    train_loader, val_loader = get_dataloaders(n_samples)
    
    # 🚀 OPTIMIZATION: Use TransformerAdapter instead of MLP
    model = TransformerAdapter(input_dim=10, hidden_dim=64, output_dim=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        experiment_name="BSOpt_Neural_Greeks_Transformer"
    )
    
    trainer.fit(train_loader, val_loader, epochs=epochs)
    return trainer.output_dir / "best_model.pt"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_neural_network()
