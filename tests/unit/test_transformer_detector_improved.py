import unittest

import numpy as np

from src.aiops.transformer_detector import TransformerAnomalyDetector


class TestTransformerDetectorReal(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.detector = TransformerAnomalyDetector(input_dim=self.input_dim, threshold=0.1)

    def test_train_and_detect(self):
        # Generate some normal training data
        train_data = np.random.randn(100, 10, self.input_dim)
        self.detector.train_on_data(train_data, epochs=5)
        self.assertTrue(self.detector.is_fitted)

        # Detect on normal data
        normal_data = np.random.randn(1, 10, self.input_dim)
        res_normal = self.detector.detect(normal_data)
        self.assertIn("is_anomaly", res_normal)

        # Detect on anomalous data (scaled up)
        anomalous_data = np.random.randn(1, 10, self.input_dim) * 10.0
        res_anom = self.detector.detect(anomalous_data)
        self.assertTrue(res_anom["score"] > res_normal["score"])

    def test_feature_attribution(self):
        train_data = np.random.randn(50, 10, self.input_dim)
        self.detector.train_on_data(train_data, epochs=2)

        # Create an anomaly in a specific feature (index 2)
        anomalous_data = np.random.randn(1, 10, self.input_dim)
        anomalous_data[0, :, 2] += 20.0

        res = self.detector.detect(anomalous_data)
        self.assertEqual(res["culprit_index"], 2)


if __name__ == "__main__":
    unittest.main()
