import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# 🚀 BORDER CONTROL: Pre-emptive mocking
class MockTensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.size = self.data.size
    def to(self, *args, **kwargs):
        return self
    def float(self):
        return self
    def dim(self):
        return len(self.shape)
    def item(self):
        return float(self.data.flatten()[0]) if self.size > 0 else 0.1
    def mean(self, *args, **kwargs):
        return MockTensor(np.mean(self.data, **kwargs))
    def numpy(self):
        return self.data
    def __sub__(self, other): 
        other_data = other.data if isinstance(other, MockTensor) else np.asarray(other)
        return MockTensor(self.data - other_data)
    def __pow__(self, other):
        return MockTensor(self.data ** other)
    def __getitem__(self, idx):
        return MockTensor(self.data[idx])
    def __len__(self):
        return len(self.data)
    def __float__(self):
        return self.item() # 🚀 Critical fix for float(mock_tensor)

# from src.aiops.transformer_detector import TransformerAnomalyDetector  # noqa: E402
# 🚀 Moved import inside class or handled via dependency injection in tests

from src.aiops.transformer_detector import TransformerAnomalyDetector


class TestTransformerDetector(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.side_effect = lambda x: x
        self.mock_model.parameters.return_value = [MockTensor([0.1])]
        self.mock_model.eval.return_value = self.mock_model
        self.mock_model.train.return_value = self.mock_model

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
        with patch("src.aiops.transformer_detector.torch.mean", return_value=MockTensor([0.05])):
            result = self.detector.detect(data)
            self.assertIn("is_anomaly", result)
            self.assertFalse(result["is_anomaly"])

    def test_train_and_detect(self):
        train_data = np.random.rand(10, 10)
        self.detector.train_on_data(train_data, epochs=1)
        self.assertTrue(self.detector.is_fitted)
        
        test_data = np.random.rand(5, 10)
        with patch("src.aiops.transformer_detector.torch.mean", return_value=MockTensor([0.05])):
            result = self.detector.detect(test_data)
            self.assertIn("is_anomaly", result)

    def test_3d_data(self):
        train_data = np.random.rand(2, 5, 10)
        self.detector.train_on_data(train_data, epochs=1)
        
        test_data = np.random.rand(1, 5, 10)
        with patch("src.aiops.transformer_detector.torch.mean", return_value=MockTensor([0.05])):
            result = self.detector.detect(test_data)
            self.assertFalse(result["is_anomaly"])

if __name__ == '__main__':
    unittest.main()
