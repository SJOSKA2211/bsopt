import pytest

import numpy as np

from src.ml.aiops.data_drift_detector import DataDriftDetector

class TestDataDriftDetector:
    def test_detect_drift_univariate(self):
        detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)
        ref = np.random.randn(100, 1)
        curr = np.random.randn(100, 1) + 2.0  # Significant drift

        drifted, info = detector.detect_drift(ref, curr)
        assert drifted
        assert "PSI" in info

    def test_detect_drift_multivariate(self):
        detector = DataDriftDetector(psi_threshold=0.1, ks_threshold=0.05)
        ref = np.random.randn(100, 3)
        curr = np.random.randn(100, 3)
        curr[:, 1] += 2.0  # Drift in feature 1

        drifted, info = detector.detect_drift(ref, curr)
        assert drifted
        assert any(d["feature_index"] == 1 for d in info["feature_drifts"])

    def test_invalid_shapes(self):
        detector = DataDriftDetector()
        with pytest.raises(ValueError):
            detector.detect_drift(np.random.randn(10, 2), np.random.randn(10, 3))

