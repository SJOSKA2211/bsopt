import unittest

import numpy as np

from src.aiops.autoencoder_detector import AutoencoderDetector


class TestAutoencoderDetector(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.latent_dim = 2
        self.detector = AutoencoderDetector(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            epochs=2,  # Keep epochs low for speed
            verbose=False,
        )

    def test_init(self):
        self.assertEqual(self.detector.input_dim, self.input_dim)
        self.assertEqual(self.detector.latent_dim, self.latent_dim)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNone(self.detector.threshold)

    def test_fit_predict(self):
        # Generate synthetic data (normal)
        data = np.random.rand(100, self.input_dim)

        self.detector.fit(data)
        self.assertIsNotNone(self.detector.threshold)

        predictions = self.detector.predict(data)
        self.assertEqual(predictions.shape, (100,))
        # Most points should be normal (1)
        self.assertTrue(np.sum(predictions == 1) > 0)

    def test_fit_empty_data_raises_error(self):
        data = np.array([])
        with self.assertRaises(ValueError):
            self.detector.fit(data)

    def test_predict_not_fitted_raises_error(self):
        data = np.random.rand(10, self.input_dim)
        with self.assertRaises(RuntimeError):
            self.detector.predict(data)

    def test_anomaly_detection(self):
        # Train on "normal" data (small values)
        normal_data = np.random.normal(0, 0.1, (100, self.input_dim))
        self.detector.fit(normal_data)

        # Test on "anomalous" data (large values)
        anomaly_data = np.random.normal(10, 1, (10, self.input_dim))
        predictions = self.detector.predict(anomaly_data)

        # Should detect anomalies (-1)
        # Note: Autoencoder might not be perfect with random weights/low epochs, but large deviation should trigger high error
        # Check if at least some are detected as -1
        self.assertTrue(np.any(predictions == -1))

    def test_fit_predict_convenience_method(self):
        data = np.random.rand(50, self.input_dim)
        predictions = self.detector.fit_predict(data)
        self.assertEqual(predictions.shape, (50,))

    def test_verbose_logging(self):
        # Create a detector with verbose=True
        verbose_detector = AutoencoderDetector(
            input_dim=self.input_dim, latent_dim=self.latent_dim, epochs=1, verbose=True
        )
        data = np.random.rand(20, self.input_dim)
        # This should print logs (captured by stdout/stderr if we checked, but here we just ensure lines are executed)
        verbose_detector.fit(data)


if __name__ == "__main__":
    unittest.main()
