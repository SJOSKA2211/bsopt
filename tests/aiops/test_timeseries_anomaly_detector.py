from unittest.mock import ANY, patch  # Import ANY

import numpy as np
import pandas as pd
import pytest

from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector


@patch("src.aiops.timeseries_anomaly_detector.logger")
class TestTimeSeriesAnomalyDetector:

    def test_init(self, mock_logger):
        """Verify that the detector initializes correctly."""
        detector = TimeSeriesAnomalyDetector(contamination=0.1)
        assert detector.model is not None
        assert detector.is_fitted is False
        mock_logger.assert_not_called()

    def test_train_success(self, mock_logger):
        """Verify that the detector trains on historical data."""
        detector = TimeSeriesAnomalyDetector()
        mock_logger.reset_mock()  # Reset logger calls after init

        # Create normal historical data (sine wave + small noise)
        t = np.linspace(0, 10, 100)
        data = pd.DataFrame(
            {
                "latency": np.sin(t) + 1.0 + np.random.normal(0, 0.1, 100),
                "cpu_usage": np.cos(t) + 1.0 + np.random.normal(0, 0.1, 100),
            }
        )

        detector.train(data)
        assert detector.is_fitted is True
        mock_logger.info.assert_called_once_with(
            "anomaly_detector_trained", samples=len(data), features=list(data.columns)
        )
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_train_empty_data(self, mock_logger):
        """Verify that training on empty data is handled gracefully."""
        detector = TimeSeriesAnomalyDetector()
        mock_logger.reset_mock()  # Reset logger calls after init

        detector.train(pd.DataFrame())
        assert not detector.is_fitted
        mock_logger.warning.assert_called_once_with("training_data_empty")
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_detect_unfitted_error(self, mock_logger):
        """Verify that detect() raises an error if model is not fitted."""
        detector = TimeSeriesAnomalyDetector()
        mock_logger.reset_mock()  # Reset logger calls after init

        with pytest.raises(
            RuntimeError, match="Model must be trained before detection."
        ):
            detector.detect(pd.DataFrame({"latency": [1.0]}))
        mock_logger.assert_not_called()

    def test_detect_empty_data(self, mock_logger):
        """Verify that detection on empty data returns an empty list."""
        detector = TimeSeriesAnomalyDetector()
        detector.train(pd.DataFrame({"latency": [1.0, 1.1, 1.2]}))  # Train it first
        mock_logger.reset_mock()  # Reset logger calls after training

        result = detector.detect(pd.DataFrame())
        assert result == []
        mock_logger.assert_not_called()

    def test_detect_anomalies(self, mock_logger):
        """Verify that the detector identifies outliers."""
        detector = TimeSeriesAnomalyDetector(contamination=0.1)
        # No need to reset mock_logger here, it's fresh for each test method due to class-level patch
        # and we only care about calls *after* this point.

        # Train on normal data
        normal_data = pd.DataFrame({"metric": np.random.normal(10, 1, 100)})
        detector.train(normal_data)

        # Reset logger calls after training for fresh assertions on detect
        mock_logger.reset_mock()

        # Detect on data with a clear anomaly
        test_data = pd.DataFrame(
            {"metric": [10.1, 9.9, 10.2, 50.0, 9.8]}  # 50.0 is an anomaly
        )

        anomalies = detector.detect(test_data)

        assert len(anomalies) >= 1
        # Check if the extreme value (index 3) was caught
        anomaly_indices = [a["index"] for a in anomalies]
        assert 3 in anomaly_indices
        assert anomalies[0]["metrics"]["metric"] == 50.0

        # Assert logger warning for the detected anomaly
        mock_logger.warning.assert_called_once_with(
            "anomaly_detected", index=ANY, score=ANY, metrics={"metric": 50.0}
        )
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
