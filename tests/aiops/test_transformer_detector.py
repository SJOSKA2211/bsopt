import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from src.aiops.transformer_detector import TransformerAnomalyDetector


class TestTransformerDetector(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        # Return a tensor that requires grad to support loss.backward()
        self.mock_model.side_effect = lambda x: x.clone().detach().requires_grad_(True)
        # Parameters must be real tensors for Adam
        self.p = nn.Parameter(torch.tensor([0.1]))
        self.mock_model.parameters.return_value = [self.p]
        self.mock_model.eval.return_value = self.mock_model
        self.mock_model.train.return_value = self.mock_model
        self.mock_model.to.return_value = self.mock_model

        with patch("src.aiops.transformer_detector.TimeSeriesTransformerEncoder", return_value=self.mock_model):
            self.detector = TransformerAnomalyDetector(input_dim=10, threshold=0.1)
        
        self.detector.scaler = MagicMock()
        self.detector.scaler.transform.side_effect = lambda x: x
        self.detector.scaler.fit.return_value = None

    def test_init(self):
        self.assertEqual(self.detector.input_dim, 10)
        self.assertEqual(self.detector.threshold, 0.1)

    def test_detect_unfitted(self):
        data = np.random.rand(5, 10)
        # Mock per_feature_loss to have 10 elements (matching input_dim)
        mock_loss = torch.full((10,), 0.05)
        with patch("src.aiops.transformer_detector.torch.mean", return_value=mock_loss):
            result = self.detector.detect(data)
            self.assertIn("is_anomaly", result)
            self.assertFalse(result["is_anomaly"])
            self.assertEqual(result["culprit_index"], 0) # argmax of all equal values is 0

    def test_train_and_detect(self):
        train_data = np.random.rand(10, 10)
        self.detector.train_on_data(train_data, epochs=1)
        self.assertTrue(self.detector.is_fitted)
        
        test_data = np.random.rand(5, 10)
        mock_loss = torch.full((10,), 0.05)
        with patch("src.aiops.transformer_detector.torch.mean", return_value=mock_loss):
            result = self.detector.detect(test_data)
            self.assertIn("is_anomaly", result)

    def test_3d_data(self):
        train_data = np.random.rand(2, 5, 10)
        self.detector.train_on_data(train_data, epochs=1)
        
        test_data = np.random.rand(1, 5, 10)
        mock_loss = torch.full((10,), 0.05)
        with patch("src.aiops.transformer_detector.torch.mean", return_value=mock_loss):
            result = self.detector.detect(test_data)
            self.assertFalse(result["is_anomaly"])

if __name__ == '__main__':
    unittest.main()
