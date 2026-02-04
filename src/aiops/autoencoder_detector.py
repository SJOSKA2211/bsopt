import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import structlog

logger = structlog.get_logger()

class VAE(nn.Module):
    """
    Variational Autoencoder for robust anomaly detection.
    Learns a latent distribution instead of fixed point embeddings.
    """
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2), # 🚀 Better for gradients
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
            nn.Sigmoid()
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

class AutoencoderDetector:
    def __init__(self, input_dim: int, latent_dim: int, epochs: int = 20,
                 threshold_percentile: float = 95.0, threshold_multiplier: float = 1.0,
                 verbose: bool = True):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        # 🚀 OPTIMIZATION: Use percentile instead of std dev for fat-tailed distributions
        self.threshold_percentile = threshold_percentile
        self.threshold_multiplier = threshold_multiplier
        self.verbose = verbose
        self.model = VAE(input_dim, latent_dim)
        self.threshold = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # KL Divergence: helps regularize the latent space
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    def fit(self, data: np.ndarray):
        if data.shape[0] == 0:
            raise ValueError("Input data for VAE must not be empty.")

        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(inputs)
                loss = self._vae_loss(recon_batch, inputs, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 5 == 0:
                logger.info("vae_epoch_complete", epoch=epoch+1, loss=total_loss/len(dataloader))
        
        # Calculate reconstruction errors to set percentile-based threshold
        self.model.eval()
        with torch.no_grad():
            reconstructions, _, _ = self.model(tensor_data)
            errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        if self.verbose:
            logger.info("anomaly_threshold_calculated", threshold=self.threshold, percentile=self.threshold_percentile)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold is None:
            raise RuntimeError("VAE model has not been fitted yet.")
        
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions, _, _ = self.model(tensor_data)
            errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
        
        # Anomalies are points with reconstruction error above the percentile-based threshold
        return np.where(errors > self.threshold, -1, 1)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.predict(data)
