import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

# ─── Safetied Torch Import ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning("torch_not_available_falling_back_to_classical_ml", error=str(e))
    TORCH_AVAILABLE = False
    # Define stubs so code doesn't crash on class definitions
    class nn:
        class Module: pass
    F = None
    TensorDataset = None
    DataLoader = None

# ─── Neural Network Components ──────────────────────────────────────────────

if TORCH_AVAILABLE:
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
else:
    VAE = None
    TimeSeriesTransformerEncoder = None


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
        
        # Determine engine availability
        if self.engine in ["autoencoder", "transformer"] and not TORCH_AVAILABLE:
            logger.warning("engine_unavailable_switching_to_isolation_forest", requested=engine)
            self.engine = "isolation_forest"

        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

        if self.engine == "isolation_forest":
            contamination = kwargs.get("contamination", 0.05)
            if not (0 < contamination <= 0.5):
                raise ValueError("Contamination must be between 0 and 0.5")
            self.model = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42)
        elif self.engine == "autoencoder":
            self.input_dim = kwargs.get("input_dim")
            self.latent_dim = kwargs.get("latent_dim", 16)
            self.threshold = None
            raw_model = VAE(self.input_dim, self.latent_dim).to(self.device)

            try:
                self.model = torch.compile(raw_model)
            except (AttributeError, Exception):
                self.model = torch.jit.script(raw_model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif self.engine == "transformer":
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

    def train(
        self, data: pd.DataFrame | np.ndarray, epochs: int = 20, study_name: str | None = None, use_ray: bool = False
    ):
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

        import mlflow
        from src.ml.tracker import ExperimentTracker

        tracker = ExperimentTracker(study_name or f"anomaly_train_{self.engine}")

        # Handle scaling
        if self.engine == "transformer" and features.ndim == 3:
            b, s, f = features.shape
            features_flat = features.reshape(-1, f)
            scaled_features = self.scaler.fit_transform(features_flat).reshape(b, s, f)
        else:
            scaled_features = self.scaler.fit_transform(features)

        if use_ray and RAY_AVAILABLE and ray.is_initialized():
            logger.info("training_anomaly_detector_with_ray", engine=self.engine)
            # Logic for Ray-based distributed training would go here
            # For IsolationForest, we can use Ray's scikit-learn adapter or train on subsets
            # For DL models, we use Ray Train
            pass

        with tracker.start_run(nested=True):
            mlflow.log_param("engine", self.engine)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("samples", len(features))
            mlflow.log_param("use_ray", use_ray)

            if self.engine == "isolation_forest":
                self.model.fit(scaled_features)

            elif self.engine == "autoencoder":
                tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
                dataloader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)
                self.model.train()
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch in dataloader:
                        inputs = batch[0]
                        self.optimizer.zero_grad()
                        recon, mu, logvar = self.model(inputs)
                        loss = nn.functional.mse_loss(
                            recon, inputs, reduction="sum"
                        ) + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()

                    mlflow.log_metric("vae_loss", epoch_loss / len(features), step=epoch)

                # Calculate threshold (95th percentile)
                self.model.eval()
                with torch.no_grad():
                    recon, _, _ = self.model(tensor_data)
                    errors = torch.mean((recon - tensor_data) ** 2, dim=1).cpu().numpy()
                    self.threshold = np.percentile(errors, 95)
                    mlflow.log_metric("vae_threshold", self.threshold)

            elif self.engine == "transformer":
                tensor_data = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
                if tensor_data.dim() == 2:
                    tensor_data = tensor_data.unsqueeze(0)

                self.model.train()
                for epoch in range(epochs):
                    self.optimizer.zero_grad()
                    recon = self.model(tensor_data)
                    loss = F.mse_loss(recon, tensor_data)
                    loss.backward()
                    self.optimizer.step()
                    mlflow.log_metric("transformer_loss", loss.item(), step=epoch)

                # Calculate threshold (95th percentile)
                self.model.eval()
                with torch.no_grad():
                    recon = self.model(tensor_data)
                    # errors: (Batch,)
                    errors = torch.mean((recon - tensor_data) ** 2, dim=(1, 2)).cpu().numpy()
                    self.threshold = np.percentile(errors, 95)
                    mlflow.log_metric("transformer_threshold", self.threshold)

        self.is_fitted = True
        logger.info("anomaly_detector_trained", engine=self.engine, samples=len(features))

    def detect(self, data: pd.DataFrame | np.ndarray, use_ray: bool = False) -> list[dict]:
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

        if use_ray and RAY_AVAILABLE and ray.is_initialized():
            # Ray-based distributed inference
            # We can parallelize the data and run 'detect' on chunks
            data_id = ray.put(scaled_features)
            model_id = ray.put(self.model)
            
            @ray.remote
            def remote_detect(chunk, model, engine, threshold):
                # Inner detection logic for a chunk
                # (Simplified for brevity)
                return [] 
            
            # Divide into 4 chunks
            chunks = np.array_split(scaled_features, 4)
            futures = [remote_detect.remote(c, model_id, self.engine, getattr(self, "threshold", None)) for c in chunks]
            # ... merge results ...
            pass

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
            with torch.inference_mode():
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

            with torch.inference_mode():
                recon = self.model(tensor_data)
                # errors: (Batch,)
                errors = torch.mean((recon - tensor_data) ** 2, dim=(1, 2)).cpu().numpy()

                for i, error in enumerate(errors):
                    if self.threshold is not None and error > self.threshold:
                        # Find culprit feature for this specific sample
                        sample_recon = recon[i]
                        sample_data = tensor_data[i]
                        per_feature_error = torch.mean((sample_recon - sample_data) ** 2, dim=0)
                        culprit_idx = int(torch.argmax(per_feature_error).item())

                        anomalies.append(
                            {
                                "index": i,
                                "is_anomaly": True,
                                "score": float(error),
                                "culprit_index": culprit_idx,
                                "type": "sequence_anomaly",
                            }
                        )

        return anomalies

    def shutdown(self):
        """Cleanup detector resources and clear GPU memory if applicable."""
        self.model = None
        self.optimizer = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("anomaly_detector_shutdown", engine=self.engine)
