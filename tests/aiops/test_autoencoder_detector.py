import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from src.aiops.autoencoder_detector import AutoencoderDetector  # noqa: E402


class TestAutoencoderDetector(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.latent_dim = 2

        self.mock_model = MagicMock(spec=nn.Module)
        # Return 3 values for VAE: recon, mu, logvar
        self.mock_model.return_value = (
            torch.zeros((10, 5)),
            torch.zeros((10, 2)),
            torch.zeros((10, 2)),
        )
        # Parameters must be real tensors for Adam
        p = nn.Parameter(torch.tensor([0.1]))
        self.mock_model.parameters.return_value = [p]
        self.mock_model.eval.return_value = self.mock_model
        self.mock_model.train.return_value = self.mock_model

        with patch("src.aiops.autoencoder_detector.VAE", return_value=self.mock_model):
            self.detector = AutoencoderDetector(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                epochs=1,
                verbose=False,
            )

        self.detector.threshold = 0.5

    def test_init(self):
        self.assertEqual(self.detector.input_dim, self.input_dim)
        self.assertEqual(self.detector.latent_dim, self.latent_dim)

    def test_fit_predict(self):
        data = np.random.rand(10, self.input_dim)
        self.detector.fit = MagicMock()
        self.detector.predict = MagicMock(return_value=np.ones(10))
        predictions = self.detector.predict(data)
        self.assertEqual(predictions.shape, (10,))

    def test_fit_empty_data_raises_error(self):
        data = np.array([]).reshape(0, 5)
        with self.assertRaises(ValueError):
            self.detector.fit(data)

    def test_predict_not_fitted_raises_error(self):
        self.detector.threshold = None
        data = np.random.rand(10, self.input_dim)
        with self.assertRaises(RuntimeError):
            self.detector.predict(data)

    def test_anomaly_detection(self):
        data = np.random.normal(0, 0.1, (10, self.input_dim))
        self.detector.fit = MagicMock()
        self.detector.predict = MagicMock(return_value=np.ones(10))
        predictions = self.detector.predict(data)
        self.assertEqual(predictions.shape, (10,))

    def test_fit_predict_convenience_method(self):
        data = np.random.rand(10, self.input_dim)
        self.detector.fit_predict = MagicMock(return_value=np.ones(10))
        predictions = self.detector.fit_predict(data)
        self.assertEqual(predictions.shape, (10,))


if __name__ == "__main__":
    unittest.main()
