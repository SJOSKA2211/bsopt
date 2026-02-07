import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector


class TestTimeSeriesAnomalyDetector(unittest.TestCase):
    def test_init(self):
        detector = TimeSeriesAnomalyDetector()
        self.assertFalse(detector.is_fitted)

    def test_train_success(self):
        detector = TimeSeriesAnomalyDetector()
        data = pd.DataFrame({"val": np.random.rand(10)})
        
        detector.model = MagicMock()
        detector.scaler = MagicMock()
        
        detector.train(data)
        self.assertTrue(detector.is_fitted)
        detector.model.fit.assert_called()

    def test_detect_anomalies(self):
        detector = TimeSeriesAnomalyDetector()
        data = pd.DataFrame({"val": np.random.rand(10)})
        detector.is_fitted = True
        
        detector.model = MagicMock()
        detector.model.predict.return_value = np.array([-1, 1, 1, -1, 1, 1, 1, 1, 1, 1])
        detector.model.decision_function.return_value = np.zeros(10)
        detector.scaler = MagicMock()
        detector.scaler.transform.side_effect = lambda x: x
        
        anomalies = detector.detect(data)
        self.assertEqual(len(anomalies), 2)
        indices = [a["index"] for a in anomalies]
        self.assertIn(0, indices)
        self.assertIn(3, indices)

    def test_train_empty_data(self):
        detector = TimeSeriesAnomalyDetector()
        data = pd.DataFrame()
        with patch("src.aiops.timeseries_anomaly_detector.logger") as mock_logger:
            detector.train(data)
            mock_logger.warning.assert_called_with("training_data_empty")

    def test_detect_unfitted_error(self):
        detector = TimeSeriesAnomalyDetector()
        data = pd.DataFrame({"a": [1]})
        with self.assertRaises(RuntimeError):
            detector.detect(data)

if __name__ == '__main__':
    unittest.main()
