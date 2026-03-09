import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger()

# ─── Neural Network Components ──────────────────────────────────────────────


class VAE(nn.Module):
    """Variational Autoencoder for robust anomaly detection."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class TimeSeriesTransformerEncoder(nn.Module):
    """Transformer-based encoder for sequential metric analysis."""

    def __init__(
        self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        return self.decoder(x)


# ─── Unified Anomaly Detector ───────────────────────────────────────────────


class AnomalyDetector:
    """
    Unified anomaly detection interface supporting multiple engines:
    - isolation_forest: Classical statistical outliers.
    - autoencoder: Deep learning based reconstruction error.
    - transformer: Sequential pattern based anomalies.
    """

    def __init__(self, engine: str = "isolation_forest", **kwargs):
        self.engine = engine
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.columns = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if engine == "isolation_forest":
            contamination = kwargs.get("contamination", 0.05)
            if not (0 < contamination <= 0.5):
                raise ValueError("Contamination must be between 0 and 0.5")
            self.model = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42)
        elif engine == "autoencoder":
            self.input_dim = kwargs.get("input_dim")
            self.latent_dim = kwargs.get("latent_dim", 16)
            self.threshold = None
            raw_model = VAE(self.input_dim, self.latent_dim).to(self.device)
            # OPTIMIZED: Use torch.compile for 2.0+ or JIT for older versions
            try:
                self.model = torch.compile(raw_model)
            except (AttributeError, Exception):
                self.model = torch.jit.script(raw_model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif engine == "transformer":
            self.input_dim = kwargs.get("input_dim")
            self.threshold = kwargs.get("threshold", 0.05)
            raw_model = TimeSeriesTransformerEncoder(self.input_dim).to(self.device)
            try:
                self.model = torch.compile(raw_model)
            except (AttributeError, Exception):
                self.model = torch.jit.script(raw_model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Unknown anomaly detection engine: {engine}")

    def train(self, data: pd.DataFrame | np.ndarray, epochs: int = 20):
        if isinstance(data, pd.DataFrame):
            numeric_df = data.select_dtypes(include=[np.number])
            self.columns = list(numeric_df.columns)
            features = numeric_df.values
        else:
            features = data
            if features.ndim == 1:
                features = features.reshape(-1, 1)

        if features.shape[0] == 0:
            return

        # Handle scaling
        if self.engine == "transformer" and features.ndim == 3:
            # Flatten for scaling: (Batch, Seq, Feat) -> (Batch*Seq, Feat)
            b, s, f = features.shape
            features_flat = features.reshape(-1, f)
            scaled_features = self.scaler.fit_transform(features_flat).reshape(b, s, f)
        else:
            scaled_features = self.scaler.fit_transform(features)

        if self.engine == "isolation_forest":
            self.model.fit(scaled_features)

        elif self.engine == "autoencoder":
            tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            dataloader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)
            self.model.train()
            for _ in range(epochs):
                for batch in dataloader:
                    inputs = batch[0]
                    self.optimizer.zero_grad()
                    recon, mu, logvar = self.model(inputs)
                    loss = nn.functional.mse_loss(
                        recon, inputs, reduction="sum"
                    ) + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss.backward()
                    self.optimizer.step()

            # Calculate threshold (95th percentile)
            self.model.eval()
            with torch.no_grad():
                recon, _, _ = self.model(tensor_data)
                errors = torch.mean((recon - tensor_data) ** 2, dim=1).cpu().numpy()
                self.threshold = np.percentile(errors, 95)

        elif self.engine == "transformer":
            tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            if tensor_data.dim() == 2:
                tensor_data = tensor_data.unsqueeze(0)

            self.model.train()
            for _ in range(epochs):
                self.optimizer.zero_grad()
                recon = self.model(tensor_data)
                loss = F.mse_loss(recon, tensor_data)
                loss.backward()
                self.optimizer.step()
            self.model.eval()

        self.is_fitted = True
        logger.info("anomaly_detector_trained", engine=self.engine, samples=len(features))

    def detect(self, data: pd.DataFrame | np.ndarray) -> list[dict]:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before detection.")

        if isinstance(data, pd.DataFrame):
            numeric_df = data.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return []
            features = numeric_df.values
        else:
            features = data
            if features.ndim == 1:
                features = features.reshape(-1, 1)

        if features.shape[0] == 0:
            return []

        # Scaling
        if self.engine == "transformer" and features.ndim == 3:
            b, s, f = features.shape
            features_flat = features.reshape(-1, f)
            scaled_features = self.scaler.transform(features_flat).reshape(b, s, f)
        else:
            scaled_features = self.scaler.transform(features)

        anomalies = []

        if self.engine == "isolation_forest":
            preds = self.model.predict(scaled_features)
            scores = self.model.decision_function(scaled_features)
            indices = np.where(preds == -1)[0]
            for idx in indices:
                anomalies.append(
                    {"index": int(idx), "score": float(scores[idx]), "type": "outlier"}
                )

        elif self.engine == "autoencoder":
            self.model.eval()
            tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                recon, _, _ = self.model(tensor_data)
                errors = torch.mean((recon - tensor_data) ** 2, dim=1).cpu().numpy()
                indices = np.where(errors > self.threshold)[0]
                for idx in indices:
                    anomalies.append(
                        {
                            "index": int(idx),
                            "score": float(errors[idx]),
                            "type": "reconstruction_error",
                        }
                    )

        elif self.engine == "transformer":
            self.model.eval()
            tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            if tensor_data.dim() == 2:
                tensor_data = tensor_data.unsqueeze(0)
            with torch.no_grad():
                recon = self.model(tensor_data)
                per_feature_loss = torch.mean((recon - tensor_data) ** 2, dim=(0, 1))
                total_loss = float(per_feature_loss.mean().item())
                if self.threshold is not None and total_loss > self.threshold:
                    culprit_idx = int(torch.argmax(per_feature_loss).item())
                    anomalies.append(
                        {
                            "is_anomaly": True,
                            "score": total_loss,
                            "culprit_index": culprit_idx,
                            "type": "sequence_anomaly",
                        }
                    )

        return anomalies
