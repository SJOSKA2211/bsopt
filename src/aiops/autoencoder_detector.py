import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # assuming data normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector:
    def __init__(self, input_dim: int, latent_dim: int, epochs: int = 10,
                 threshold_multiplier: float = 2.0, verbose: bool = True):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.threshold_multiplier = threshold_multiplier
        self.verbose = verbose
        self.model = Autoencoder(input_dim, latent_dim)
        self.threshold = None
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, data: np.ndarray):
        if data.shape[0] == 0:
            raise ValueError("Input data for Autoencoder must not be empty.")

        # Convert numpy array to PyTorch Tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Calculate reconstruction errors on the training data to set the threshold
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(tensor_data)
            errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
        self.threshold = np.mean(errors) + self.threshold_multiplier * np.std(errors)
        if self.verbose:
            print(f"Calculated anomaly threshold: {self.threshold:.4f}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold is None:
            raise RuntimeError("Autoencoder model has not been fitted yet.")
        
        # Convert numpy array to PyTorch Tensor
        tensor_data = torch.tensor(data, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(tensor_data)
            errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
        
        # Anomalies are points with reconstruction error above the threshold
        return np.where(errors > self.threshold, -1, 1)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.predict(data)
