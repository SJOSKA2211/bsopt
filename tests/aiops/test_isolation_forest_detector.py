import unittest
import numpy as np
from src.aiops.isolation_forest_detector import IsolationForestDetector

class TestIsolationForestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = IsolationForestDetector(contamination=0.1)

    def test_init_invalid_contamination(self):
        with self.assertRaises(ValueError):
            IsolationForestDetector(contamination=0.6)
        with self.assertRaises(ValueError):
            IsolationForestDetector(contamination=0.0)

    def test_fit_predict_1d(self):
        data = np.random.rand(100)
        predictions = self.detector.fit_predict(data)
        self.assertEqual(predictions.shape, (100,))
        self.assertIsNotNone(self.detector.model)

    def test_fit_predict_2d(self):
        data = np.random.rand(100, 2)
        predictions = self.detector.fit_predict(data)
        self.assertEqual(predictions.shape, (100,))

    def test_fit_predict_empty_data(self):
        data = np.array([])
        with self.assertRaises(ValueError):
            self.detector.fit_predict(data)

    def test_predict_not_fitted(self):
        data = np.random.rand(10)
        with self.assertRaises(RuntimeError):
            self.detector.predict(data)

    def test_predict_success(self):
        train_data = np.random.rand(100, 1)
        self.detector.fit_predict(train_data)
        
        test_data = np.random.rand(10, 1)
        predictions = self.detector.predict(test_data)
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))

    def test_predict_1d_reshape(self):
        train_data = np.random.rand(100, 1)
        self.detector.fit_predict(train_data)
        
        test_data = np.random.rand(10)
        predictions = self.detector.predict(test_data)
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
