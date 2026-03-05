"""
TransformerAnomalyDetector — sequence-based anomaly detection using a compact
Transformer autoencoder.

The model treats time-series windows of shape (batch, seq_len, input_dim) as
input, encodes them through multi-head self-attention + FFN layers, and
reconstructs them.  The reconstruction error (MSE per feature, averaged over
the sequence) is used as the anomaly score.

Designed to run on CPU without heavy dependencies; PyTorch is used only during
training / inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class TransformerAnomalyDetector:
    """
    Lightweight Transformer-autoencoder anomaly detector.

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    d_model : int
        Transformer internal embedding dimension (default 32).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of TransformerEncoder layers (default 2).
    threshold : float
        Reconstruction MSE above which a window is labelled anomalous.
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        threshold: float = 0.5,
    ) -> None:
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.threshold = threshold
        self.is_fitted = False
        self._model: Any = None  # populated in train_on_data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_data(
        self,
        data: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """
        Train the Transformer autoencoder on *data*.

        Parameters
        ----------
        data : np.ndarray, shape (n_windows, seq_len, input_dim)
        epochs : int
        lr : float
        batch_size : int
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for TransformerAnomalyDetector.train_on_data"
            ) from exc

        X = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = _TransformerAutoencoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

        self._model = model
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, window: np.ndarray) -> dict[str, Any]:
        """
        Compute reconstruction error for *window* and return anomaly info.

        Parameters
        ----------
        window : np.ndarray, shape (1, seq_len, input_dim) or (seq_len, input_dim)

        Returns
        -------
        dict with keys:
            - ``is_anomaly`` (bool)
            - ``score``      (float, per-sample MSE)
            - ``culprit_index`` (int | None, feature with highest mean error)
            - ``feature_errors``  (list[float])
        """
        if not self.is_fitted or self._model is None:
            # Return a null response if not yet trained (graceful degradation)
            return {
                "is_anomaly": False,
                "score": 0.0,
                "culprit_index": None,
                "feature_errors": [],
            }

        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch required for detect()") from exc

        arr = np.atleast_3d(window)
        if arr.shape[0] != 1:
            arr = arr[np.newaxis, ...]  # type: ignore[assignment]

        x = torch.tensor(arr, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(x)

        error = (x - reconstructed) ** 2  # shape: (1, seq_len, input_dim)
        feature_errors = error[0].mean(dim=0).numpy().tolist()  # mean over time
        score = float(error.mean().item())

        culprit_index = int(np.argmax(feature_errors)) if feature_errors else None

        return {
            "is_anomaly": score > self.threshold,
            "score": score,
            "culprit_index": culprit_index,
            "feature_errors": feature_errors,
        }


# ---------------------------------------------------------------------------
# Internal PyTorch model (defined here to avoid circular imports)
# ---------------------------------------------------------------------------

class _TransformerAutoencoder:
    """Lazy-imported after torch is confirmed present."""

    def __new__(cls, input_dim: int, d_model: int, nhead: int, num_layers: int) -> Any:  # type: ignore[override]
        try:
            import torch.nn as nn

            class _Inner(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=d_model * 4,
                        batch_first=True,
                        dropout=0.0,
                    )
                    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.decoder = nn.Linear(d_model, input_dim)

                def forward(self, x):  # type: ignore[override]
                    emb = self.input_proj(x)
                    enc = self.encoder(emb)
                    return self.decoder(enc)

            return _Inner()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch required") from exc
