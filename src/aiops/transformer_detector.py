from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


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
    """
    Advanced Transformer-based anomaly detector with feature-level attribution.
    Uses reconstruction error per feature to identify root causes.
    """
    def __init__(
        self,
        input_dim: int,
        feature_names: list[str] | None = None,
        threshold: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.input_dim = input_dim
        self.model = TimeSeriesTransformerEncoder(input_dim).to(device)
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]
        self.threshold = threshold
        self.device = device
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.model.eval()

    def detect(self, data: np.ndarray) -> dict[str, Any]:
        """
        Detects anomalies and identifies the primary feature contributor.
        """
        if not self.is_fitted:
            # Fallback to no scaling if not yet fitted
            x_scaled = data
        else:
            # Ensure 2D for scaler
            original_shape = data.shape
            if len(original_shape) == 3:
                # (Batch, Seq, Feat) -> Flatten batch/seq for scaling
                data_flat = data.reshape(-1, original_shape[-1])
                x_scaled = self.scaler.transform(data_flat).reshape(original_shape)
            else:
                x_scaled = self.scaler.transform(data)

        with torch.no_grad():
            x = torch.from_numpy(x_scaled).float().to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(0) # Add batch dim
            
            reconstructed = self.model(x)
            
            # 🚀 Rick Optimization: Per-feature error attribution
            # Mean squared error for each feature across the batch and window
            per_feature_loss = torch.mean((reconstructed - x)**2, dim=(0, 1))
            total_loss = per_feature_loss.mean().item()
            
            culprit_idx = torch.argmax(per_feature_loss).item()
            culprit_name = self.feature_names[culprit_idx]
            
            is_anomaly = total_loss > self.threshold
            
            return {
                "is_anomaly": is_anomaly,
                "score": total_loss,
                "culprit_index": culprit_idx,
                "culprit_name": culprit_name,
                "feature_errors": {self.feature_names[i]: float(per_feature_loss[i]) for i in range(len(per_feature_loss))}
            }

    def train_on_data(self, train_data: np.ndarray, epochs: int = 50):
        """
        Trains the model and fits the internal scaler.
        """
        self.model.train()
        
        # Fit and apply scaling
        original_shape = train_data.shape
        if len(original_shape) == 3:
            data_flat = train_data.reshape(-1, original_shape[-1])
            self.scaler.fit(data_flat)
            scaled_data = self.scaler.transform(data_flat).reshape(original_shape)
        else:
            self.scaler.fit(train_data)
            scaled_data = self.scaler.transform(train_data)
            
        self.is_fitted = True
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        x = torch.from_numpy(scaled_data).float().to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = F.mse_loss(reconstructed, x)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
        
        self.model.eval()
