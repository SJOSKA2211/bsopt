import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) # Max seq len 1000
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x

class TransformerAnomalyDetector:
    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = TimeSeriesTransformerEncoder(input_dim).to(device)
        self.threshold = threshold
        self.device = device
        self.model.eval()

    def detect(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Detects anomalies based on reconstruction error.
        """
        with torch.no_grad():
            x = torch.from_numpy(data).float().to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(0) # Add batch dim
            
            reconstructed = self.model(x)
            loss = F.mse_loss(reconstructed, x).item()
            
            is_anomaly = loss > self.threshold
            return is_anomaly, loss

    def train_on_data(self, train_data: np.ndarray, epochs: int = 50):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        x = torch.from_numpy(train_data).float().to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = F.mse_loss(reconstructed, x)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
        
        self.model.eval()
